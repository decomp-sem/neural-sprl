import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import sys
import logging

class SPRNet(nn.Module):
  """
  Network component that predicts binary or scalar values for
  each of <how many?> semantic proto-role (SPR) attributes,
  either from SPR v1.0 or SPR v2.0. Input is a single vector,
  in practice, the concatenation of two hidden states: one
  corresponding to the predicate head, the other corresponding
  to the argument head. (SPR labels are wrt to a pred-arg pair.)
  """

  def __init__(self, input_dim=1200, spr2=False, regr=False,
    num_classes=2, gpu=False):
    """
    Keyword Args
      input_dim: (int) The size of the (concatenated) input vector.
      spr2: (bool) True iff SPR v2.0; else SPR v1.0. Determines the set of
        attributes used.
      regr: (bool) Whether the prediction is scalar-valued with regression (regr==True),
        or over discrete classes with a softmax layer (regr==False). Default is False.
      num_classes: Number of classes if discrete classification. Default is 2.
    """

    # define the functional components of the network with trainable params
    super(SPRNet, self).__init__()

    # Check argument validity.
    if spr2:
      assert regr, "SPR2 requires regression; SPR1 can be classification or regr."
    if num_classes != 2:
      raise NotImplementedError, "Discrete SPR prediction only implemented for num_classes==2."

    self.input_dim = input_dim
    self.regr = regr
    self.num_classes = num_classes
    self.gpu = gpu
    
    # input size to the attribute-specific predictions
    self.attr_input_dim = self.input_dim
    if spr2:
      self.spr_attributes = [u'awareness',u'change_of_location',
        u'change_of_possession',u'change_of_state',
        u'change_of_state_continuous',u'existed_after',u'existed_before',
        u'existed_during',u'instigation',u'partitive',u'sentient',u'volition',
        u'was_for_benefit',u'was_used']
    else:
      self.spr_attributes = [u'awareness',u'change_of_location',u'change_of_state',
        u'changes_possession',u'existed_after',u'existed_before',u'existed_during',
        u'exists_as_physical',u'instigation',u'location_of_event',u'makes_physical_contact',
        u'manipulated_by_another',u'predicate_changed_argument',u'sentient',u'stationary',
        u'volition',u'created',u'destroyed']
      
    # shared layer
    self.lin_shared = torch.nn.Linear(self.input_dim, self.attr_input_dim)
    self.relu = torch.nn.ReLU()

    if regr:
      self.attr_hidden_dim = int(self.attr_input_dim/2)
      for attr in self.spr_attributes:
        # 2-layer version
        #self.add_module("lin1_attr_"+attr, torch.nn.Linear(self.attr_input_dim, self.attr_hidden_dim))
        #self.add_module("lin2_attr_"+attr, torch.nn.Linear(self.attr_hidden_dim, 1))
        # 1-layer version
        self.add_module("lin_attr_"+attr, torch.nn.Linear(self.attr_input_dim, 1))

    else: # discrete prediction
      self.logsm = torch.nn.LogSoftmax()
      for attr in self.spr_attributes:
        # 1-layer version
        self.add_module("lin_attr_"+attr, torch.nn.Linear(self.attr_input_dim, self.num_classes))
        # 2-layer version
        #self.attr_hidden_dim = int(self.attr_input_dim/2)
        #self.add_module("lin1_attr_"+attr, torch.nn.Linear(self.attr_input_dim, self.attr_hidden_dim))
        #self.add_module("lin2_attr_"+attr, torch.nn.Linear(self.attr_hidden_dim, self.num_classes))

  def forward(self, x, **kwargs):
    """
    Inputs
      x: A vector of size input_dim.
    """
   
    # First apply a linear transformation followed by ReLU, with shared
    # params for all attributes
    x = self.lin_shared(x)
    x = self.relu(x)

    spr_out = {}
    if self.regr:
      for attr in self.spr_attributes:
        # 2-layer version
        #spr_out[attr] = self._modules["lin1_attr_"+attr](x) # first layer of MLP
        #spr_out[attr] = self.relu(spr_out[attr]) # MLP nonlinearity
        #spr_out[attr] = self._modules["lin2_attr_"+attr](spr_out[attr]) # final layer of MLP, output is scalar
        # 1-layer version
        spr_out[attr] = self._modules["lin_attr_"+attr](x) # first layer of MLP
    else: # discrete
      for attr in self.spr_attributes:
        # 1-layer version
        spr_out[attr] = self._modules["lin_attr_"+attr](x) # pre-softmax linear transformation
        spr_out[attr] = self.logsm(spr_out[attr])
        # 2-layer version
        #spr_out[attr] = self._modules["lin1_attr_"+attr](x) # pre-softmax linear transformation
        #spr_out[attr] = self.relu(spr_out[attr]) # MLP nonlinearity
        #spr_out[attr] = self._modules["lin2_attr_"+attr](spr_out[attr]) # pre-softmax linear transformation
        #spr_out[attr] = self.logsm(spr_out[attr])

    return spr_out

  def loss(self, y, yhat):
    """
    Inputs:
      y: Dict mapping spr properties to actual label (scalar or binary.)
      yhat: Dict mapping from spr properties to either scalar prediction or
         binary softmax.
    """
    ylosses = {}
    yloss_total = 0.0
    
    for attr in self.spr_attributes:
      assert (attr in y) and (attr in yhat)
    if self.regr:
      for attr in self.spr_attributes:
        y_attr = 1.0 if math.isnan(y[attr]) else y[attr]
        if self.gpu:
          y_attr_tensor = Variable(torch.from_numpy(np.asarray([y_attr])).cuda().float())
        else:
          y_attr_tensor = Variable(torch.from_numpy(np.asarray([y_attr])).float())
        ylosses[attr] = torch.nn.functional.mse_loss(yhat[attr], y_attr_tensor)
        yloss_total += ylosses[attr]
    else:
      # loss is negative log probability of correct class, per property
      for attr in self.spr_attributes:
        # y_attr_tensor is the binary value (0,1) of attribute applicability;
        # it is also the index into the corresponding logsoftmax distribution
        ylosses[attr] = yhat[attr][y[attr]]
        yloss_total += -1.0 * ylosses[attr]
    return ylosses, yloss_total
