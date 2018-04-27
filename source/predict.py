import sys, io
import json, csv
import torch
import numpy as np
import logging
import argparse
from multitask_net import MultiNet
from util import load_stanford
#from task_utils import *

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', action='store_true', default=False,
                      help="Run on GPU.")
  parser.add_argument('--spr_version', action='store', type=str,
                      choices=["spr1c","spr2"],
                      default="spr1c",
                      help="Which version of SPR labeling to run.")
  parser.add_argument('--load_torch', action='store', type=str, default=None,
                      help="Full path to file from which to LOAD a saved pytorch model;\
                      may be used in either train or score mode.")
  parser.add_argument('--input_file', action='store', type=str, default="../data/mini.sprl",
                      help="Full path to (read-only) file with input data to annotate.")
  parser.add_argument('--output_file', action='store', type=str, default="../data/mini.sprl.out",
                      help="Full path to file to write labeled output to.")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  if args.gpu:
    # claim the available gpu
    claim_gpu = torch.FloatTensor(1).cuda()

  embedding_path="../models/vocab.txt"

  logging.info("Loading glove vectors, may take a few minutes...")
  wtoi, itow, word_vectors = load_stanford(embedding_path)
  logging.info("Glove vectors successfully loaded.")

  # compute model parameters from embedding size
  num_embeddings_glove, embedding_dim_glove = word_vectors.shape
  embedding_weights_glove = word_vectors

  spr_version = args.spr_version
  if spr_version == "spr1c":
    load_torch_model_file = "../models/spr1c_torch_model"
  if spr_version == "spr2":
    load_torch_model_file = "../models/spr2_torch_model"
  logging.info("Loading torch model from %s..." % load_torch_model_file)
  MN = torch.load(load_torch_model_file)

  # determine gpu settings
  MN.set_gpu(args.gpu)
  if args.gpu:
    MN.cuda()

  def wtoi_func(wtoi, w):
    UNK = u'<unk>'
    try:
      return wtoi[w]
    except KeyError:
      return wtoi[UNK]

  def convert_predictions(all_predictions):
    # in-place conversion that converts output predictions from the model
    # from torch Variables to python floats (for json serialization)
    # also convert tuple keys to string keys
    for pred_arg_idx in all_predictions.keys():
      spr_predictions = all_predictions[pred_arg_idx]
      all_predictions[str(pred_arg_idx)] = spr_predictions
      all_predictions.pop(pred_arg_idx)
      for spr_property in spr_predictions.keys():
        if spr_version == "spr2":
          spr_predictions[spr_property] = round(float(spr_predictions[spr_property].data.numpy()[0]), 4)
        if spr_version == "spr1c":
          spr_predictions[spr_property] = round(float(spr_predictions[spr_property].data.numpy()[1]), 4)

  input_file = io.open(args.input_file, "r", encoding="utf-8")
  output_file = io.open(args.output_file, "w", encoding="utf-8")

  for line in input_file:
    json_obj = json.loads(line, encoding="utf-8")

    # pre-process sentence input
    toks = json_obj["tokens"]
    toks_lower = map(lambda x: x.lower(),toks)
    tok_ids = map(lambda x: wtoi_func(wtoi,x), toks_lower)
    pred_args = [tuple(x) for x in json_obj["pred-args"]]

    task2idxs = {spr_version: pred_args}
    outputs = MN(tok_ids, task2idxs)
    json_obj["predictions"] = outputs[spr_version]
    convert_predictions(json_obj["predictions"])
    json_out_str = json.dumps(json_obj, encoding="utf-8")
    output_file.write(u""+json_out_str+"\n")

  input_file.close()
  output_file.close()
