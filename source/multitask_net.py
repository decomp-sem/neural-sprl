import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

from spr_net import SPRNet
from sense_net import SenseNet
from role_net import RoleNet

class MultiNet(nn.Module):
  """
  Multitask network for predicting decompositional semantics labels.
  """

  def __init__(self, input_size=300, hidden_size=300, bidirectional=True,
    layers=1, tasks={"spr1c","spr1r","spr2","supersense","propbank","framenet"},
    supersenses=None,
    num_embeddings=1001, embedding_dim=300, embedding_weights=None, gpu=False):
    """
    Keyword Args
      input_size:  The dimension of the input embeddings. (default 300)
      hidden_size:  The hidden size of the LSTM (in one direction). (default 300.)
      bidirectional:  True iff the LSTM(s) are bidirectional. (default True)
      layers:  The number of layers in (each) LSTM, not including bidirectionality. (default 1)
      tasks:  A set of names (str), one for each data source/task in the multitask setting.
        Default (all tasks): {"spr1c","spr1r","spr2","supersense","propbank","framenet"}
      num_embeddings:  The vocabulary size/number of embeddings. (default 1001, though this value
        should always be provided by the user.)
      embedding_dim:  The dimension of the input embeddings. (default 300)
      embedding_weights:  The actual embedding matrix, e.g. pre-trained glove embeddings or a
        randomly-initialized embedding matrix. Code has only been tested with using pre-trained
        embeddings that do not update during training. np.ndarray
      gpu:  True iff training on gpu. Default False.
    """

    # define the functional components of the network with trainable params
    super(MultiNet, self).__init__()

    # set hardware params
    self.gpu = gpu

    # set network hyperparameters
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.layers = layers

    # task-specific information
    self.supersenses = supersenses

    # set embedding hyperparameters
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    assert self.embedding_dim == self.input_size

    # multitask hyperparameters
    self.tasks = tasks

    # size attributes computed from other network hyperparameters
    self.total_layers = self.layers * 2 if self.bidirectional else self.layers
    self.output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

    # define embedding layer
    self.embeddings = nn.Embedding(num_embeddings, embedding_dim,
      padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False,
      sparse=False)

    assert embedding_weights is not None, "User must provide embedding weights to Net."

    # requires_grad is False, because the pretrained embeddings are FIXED
    self.embeddings.weight = nn.Parameter(torch.FloatTensor(embedding_weights),
      requires_grad=False)

    assert self.embeddings.weight.size() == torch.Size([num_embeddings, embedding_dim]), "\
      Provided embedding weights must be of dim [num_embeddings x embedding_dim]."
 
    # define L-BiLSTM RNN
    self.rnn = torch.nn.LSTM(self.input_size, self.hidden_size,
      num_layers=self.layers, bidirectional=self.bidirectional) 
  
    # define initial hidden states as learnable model parameters
    self.h0_param = torch.nn.Parameter(torch.randn(self.total_layers, 1,
      self.hidden_size))
    self.c0_param = torch.nn.Parameter(torch.randn(self.total_layers, 1,
      self.hidden_size))

    # define the multiple tasks of the network
    # Not all instantiations of the class will have all submodules for all tasks;
    # this makes sense because at test time, you wouldn't want to try doing a task
    # that the model hadn't trained on.
    self.all_possible_tasks = ["spr1c","spr1r","spr2","supersense","propbank","framenet"]

    # Relevant info
    # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113

    ############################
    # ADD MODULE FOR EACH TASK #
    ############################

    # SPR1.0 task, classification
    if "spr1c" in self.tasks:
      # The input is a concatenated pair of bilstm hidden states
      self.SPR1C = SPRNet(input_dim=2*self.output_size)

    # SPR1.0 task, regression
    if "spr1r" in self.tasks:
      # The input is a concatenated pair of bilstm hidden states
      if self.gpu:
        self.SPR1R = SPRNet(input_dim=2*self.output_size, regr=True, gpu=self.gpu)
      else:
        self.SPR1R = SPRNet(input_dim=2*self.output_size, regr=True)

    # SPR2.0 task
    if "spr2" in self.tasks:
      # The input is a concatenated pair of bilstm hidden states
      if self.gpu:
        self.SPR2 = SPRNet(input_dim=2*self.output_size,spr2=True, regr=True, gpu=self.gpu)
      else:
        self.SPR2 = SPRNet(input_dim=2*self.output_size,spr2=True, regr=True)

    # Supersense task
    if "supersense" in self.tasks:
      self.Sense = SenseNet(input_dim=self.output_size, supersenses=self.supersenses)

    # Propbank task
    if "propbank" in self.tasks:
      # Set transform to True, so that this module can be distinct from
      # the framenet module.
      self.Propbank = RoleNet(input_dim=2*self.output_size, roles=None,
        is_propbank=True, transform=True)

    # Framenet task
    if "framenet" in self.tasks:
      # TODO not implemented, put in real framenet roles
      # Set transform to True so that this module can be distinct from
      # the propbank module
      self.Framenet = RoleNet(input_dim=2*self.output_size, roles=None,
        is_propbank=False, transform=True)

  @staticmethod
  def get_pair(index_pair, states):
    """
    Inputs:
      index_pair: A pair of sequence indexes into states; the first
        corresponds to the PREDICATE head token, the second corresponds
        to the ARGUMENT head token.
      states: final layer of states from bilstm [seq_len, state_dim]
    """
    idx1, idx2 = index_pair
    s1, s2 = states[idx1,:], states[idx2,:]
    return torch.cat((s1,s2))

  def forward(self, x, tasks2idxs, **kwargs):
    """
    Inputs
      x: Tensor of token indices
      tasks2idxs: Dict mapping from task name (matches keys in self.tasks) to
        some (task-specific) object/iterable with info about which token id(s)
        will be used as input to that tasks's prediction submodule.
    """

    if self.gpu:
      x = Variable(torch.LongTensor(x).cuda())
    else:
      x = Variable(torch.LongTensor(x))

    assert all(k in self.tasks for k in tasks2idxs.keys()), "tasks2idxs keys must be subset of self.tasks."

    # prepare inputs
    x_emb = self.embeddings(x)
    x_emb = x_emb.unsqueeze(1) # (seq_len, input_size) --> (seq_len, batch==1, input_size)
    # initial RNN states will be set/overwritten after MultiNet is initialized.
    # pass through rnn
    states, (h_n, c_n) = self.rnn(x_emb, (self.h0_param, self.c0_param))
    states = states.squeeze(1) # squeeze out batch==1 dimension

    # initialize output dict
    out = {}

    # select appropriate inputs for each task
    # SPR 1.0 classification
    if "spr1c" in tasks2idxs:
      spr1c_out = {}
      for idx in tasks2idxs["spr1c"]:
        state_pair_input = MultiNet.get_pair(idx, states)
        spr1c_out[idx] = self.SPR1C(state_pair_input)
      out["spr1c"] = spr1c_out

    # SPR 1.0 regression
    if "spr1r" in tasks2idxs:
      spr1r_out = {}
      for idx in tasks2idxs["spr1r"]:
        state_pair_input = MultiNet.get_pair(idx, states)
        spr1r_out[idx] = self.SPR1R(state_pair_input)
      out["spr1r"] = spr1r_out
     
    # SPR 2.0
    if "spr2" in tasks2idxs:
      spr2_out = {}
      for idx in tasks2idxs["spr2"]:
        state_pair_input = MultiNet.get_pair(idx, states)
        spr2_out[idx] = self.SPR2(state_pair_input)
      out["spr2"] = spr2_out

    # Supersense task
    if "supersense" in tasks2idxs:
      supersense_out = {}
      for idx in tasks2idxs["supersense"]:
        state = states[idx,:]
        supersense_out[idx] = self.Sense(state)
      out["supersense"] = supersense_out

    # Propbank task
    if "propbank" in tasks2idxs:
      propbank_out = {}
      for idx in tasks2idxs["propbank"]:
        state_pair_input = MultiNet.get_pair(idx, states)
        propbank_out[idx] = self.Propbank(state_pair_input)
      out["propbank"] = propbank_out

    # Framenet task
    if "framenet" in tasks2idxs:
      framenet_out = {}
      for idx in tasks2idxs["framenet"]:
        state_pair_input = MultiNet.get_pair(idx, states)
        framenet_out[idx] = self.Framenet(state_pair_input)
      out["framenet"] = framenet_out

    return out

  def set_gpu(self, gpu):
    """ gpu is boolean """
    self.gpu = gpu
    if "spr1r" in self.tasks:
      self.SPR1R.gpu = gpu
    if "spr2" in self.tasks:
      self.SPR2.gpu = gpu
    
  def loss(self, labels, predictions):
    """
    Inputs:
      
    """
    assert all(k in labels.keys() for k in predictions.keys())
    assert all(k in predictions.keys() for k in labels.keys())

    task2losses = {}
    total_loss = 0.0

    def compute_submodule_loss(submodule, submodule_key):
      submodule_loss = {}
      submodule_total = 0.0
      for idx, label in labels[submodule_key].iteritems():
        prediction = predictions[submodule_key][idx]
        if submodule_key in ["spr1c","spr1r","spr2","supersense"]:
          instance_losses, total_instance_loss = submodule.loss(label, prediction)
          submodule_loss[idx] = (total_instance_loss, instance_losses)
          submodule_total += total_instance_loss
        else:
          loss = submodule.loss(label, prediction)
          submodule_loss[idx] = loss
          submodule_total += loss
      return submodule_loss, submodule_total

    if "spr1c" in labels.keys():
      spr1c_losses, spr1c_total = compute_submodule_loss(self.SPR1C, "spr1c")
      task2losses["spr1c"] = spr1c_losses
      total_loss += spr1c_total
    if "spr1r" in labels.keys():
      spr1r_losses, spr1r_total = compute_submodule_loss(self.SPR1R, "spr1r")
      task2losses["spr1r"] = spr1r_losses
      total_loss += spr1r_total
    if "spr2" in labels.keys():
      spr2_losses, spr2_total = compute_submodule_loss(self.SPR2, "spr2")
      task2losses["spr2"] = spr2_losses
      total_loss += spr2_total
    if "supersense" in labels.keys():
      supersense_losses, supersense_total = compute_submodule_loss(self.Sense, "supersense")
      task2losses["supersense"] = supersense_losses
      total_loss += supersense_total
    if "propbank" in labels.keys():
      propbank_losses, propbank_total = compute_submodule_loss(self.Propbank, "propbank")
      task2losses["propbank"] = propbank_losses
      total_loss += propbank_total
    if "framenet" in labels.keys():
      framenet_losses, framenet_total = compute_submodule_loss(self.Framenet, "framenet")
      task2losses["framenet"] = framenet_losses
      total_loss += framenet_total

    return task2losses, total_loss
