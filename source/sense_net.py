import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

class SenseNet(nn.Module):
  """
  Network component that predicts one of 41 possible discrete
  WordNet supersense tags. Input is bilstm hidden state (or
  possibly concatenation of two hidden states).
  Information about the 41 supersense tags are here:
  https://dl.acm.org/citation.cfm?id=1610158
  """

  def __init__(self, input_dim=600, supersenses=None, transform=False):
    """
    Keyword Args
      input_dim: (int) The size of the (concatenated) input vector.
      supersenses: (list) A list of all (41) possible supersense tags (str).
        The order in the list corresponds to the order of the softmax/
        probability outputs of the network.
      transform: (bool) Whether to perform a linear+ReLU transofmration on
        the input before performing classification. This transformation
        preserves the original dimensionality of the input.
    """

    # define the functional components of the network with trainable params
    super(SenseNet, self).__init__()

    self.input_dim = input_dim
    self.supersenses = supersenses
    self.stoi = {s: i for i, s in enumerate(self.supersenses)}
    if self.supersenses is None:
      self.supersenses = ["animal","event","quantity","time"] # examples for development
    self.num_classes = len(self.supersenses)
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
      y: Correct (discrete) WordNet supersense. (integer id)
      yhat: Prediction for WordNet supersense. (Softmax dist.)
    """
    ylosses = {}
    yloss_weighted_total = 0.0
    for sense in y.keys():
      sense_idx = self.stoi[sense]
      sense_weight = y[sense]
      ylosses[sense] = yhat[sense_idx]
      yloss_weighted_total += -1.0 * sense_weight * yhat[sense_idx]
    return ylosses, yloss_weighted_total
