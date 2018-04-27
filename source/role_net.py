import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import logging

class RoleNet(nn.Module):
  """
  Network component that predicts one of <N> possible discrete
  role labels, as defined either by Propbank or FrameNet.
  Input is a single vector, in practice, the concatenation of
  two hidden states: one corresponding to the predicate head, the
  other corresponding to the argument head. This is analogous to
  the input of the SPR net.
  For Propbank, we predict the new labels that have better
  correspondence across frames, e.g., "PAG," "PPT," "GOL,"...
  instead of the old "ARG0," "ARG1,"... style arg labels.
  For Framenet, the precise Frame Element label set is TBD...
  We use one module for these two tasks because we are using
  the same architecture for both tasks.
  """

  def __init__(self, input_dim=600, roles=None, is_propbank=True, transform=False):
    """
    Keyword Args
      input_dim: (int) The size of the (concatenated) input vector. Default 600.
      roles: (list) A list of all <N> possible role labels, from either
        propbank or framenet. The order in the list corresponds to the order
        of the softmax/probability outputs of the network.
      is_propbank: (bool) True iff prediction is for propbank labels; False
        iff prediction is for framenet labels. Default is True.
      transform: (bool) Whether to perform a linear+ReLU transofmration on
        the input before performing classification. This transformation
        preserves the original dimensionality of the input.
    """

    # define the functional components of the network with trainable params
    super(RoleNet, self).__init__()

    self.input_dim = input_dim
    self.is_propbank = is_propbank
    self.roles = roles

    if self.roles is None:
      if self.is_propbank:
        #self.roles = ["PPT","PAG","GOL"] # examples for development
        self.roles = ['adj','adv','cau','com','dir','ext','gol','loc','mnr','pag','ppt','prd','prp','rec','tmp','vsp']
      else:
        self.roles = ["avenger","offender","injury"] # examples for development

    self.rtoi = {r: i for i, r in enumerate(self.roles)}
    self.num_classes = len(self.roles)
    self.transform = transform

    if not self.transform:
      self.transformed_input_dim = self.input_dim
    else:
      # Currently, if a pre-classification transformation is performed, it's resulting
      # dimensionality is the same as the original input anyway.
      self.transformed_input_dim = self.input_dim


    # optional pre-classification transformation
    if self.transform:
      self.lin_t = torch.nn.Linear(self.input_dim, self.transformed_input_dim)
      # ReLU is applied only if transform==True
      self.relu = torch.nn.ReLU()
      
    # rest of classification network, after the initial (optional) transformation
    self.lin = torch.nn.Linear(self.transformed_input_dim, self.num_classes)
    self.logsm = torch.nn.LogSoftmax()


  def forward(self, x, **kwargs):
    """
    Inputs
      x: A vector of size input_dim.
    """
    if self.transform:
      # First apply an optional linear transformation followed by ReLU, with shared
      # params for all attributes, if transform==True
      x = self.lin_t(x)
      x = self.relu(x)

    x = self.lin(x) # pre-softmax linear transformation
    x = self.logsm(x)

    return x

  def loss(self, y, yhat):
    """
    Inputs:
      y: Correct (discrete) Propbank or Framenet role label. (integer id?)
      yhat: Prediction for Propbank or Framenet role label. (Softmax dist.)
    """
    if self.is_propbank:
      y_ind = self.rtoi[y]
      yloss = -1.0*yhat[y_ind]
    else:
      yloss = -1.0*yhat[y]
      raise NotImplementedError
    return yloss
