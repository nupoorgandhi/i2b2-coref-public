#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import tensorflow as tf
import util

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  # tf.enable_eager_execution()

  config = util.initialize_from_env()

  dataset = sys.argv[3]

  if dataset == 'tgt':
      config['eval_path'] = '/projects/tir4/users/nmgandhi/coref/data/jsonlines/test.i2b2.512.jsonlines'
      # config['eval_path'] = '/projects/tir4/users/nmgandhi/coref/data/jsonlines/test.i2b2.english.512.jsonlines'
      config['conll_eval_path'] = '/projects/tir4/users/nmgandhi/coref/data/conll/test.i2b2.conll'
  elif dataset == 'src':
      config['eval_path'] = '/projects/tir4/users/nmgandhi/coref/data/jsonlines/test.english.512.jsonlines'
      config['conll_eval_path'] = '/projects/tir4/users/nmgandhi/coref/data/conll/test.english.v4_gold_conll'
  else:
      print('need to specify the dataset')
      exit(0)
  print('config[model_type]', config['model_type'])
  if config['model_type'] == 'projection' or  config['model_type'] == 'noncontext' or config['model_type'] == 'joint' :
      print('identified')
      projection_args = util.initialize_yaml_from_env(config['projection_yaml'])

      model = util.get_model(config, projection_args)
  else:
      # exit(0)
      model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  with tf.Session() as session:
    model.restore(session)
    # Make sure eval mode is True if you want official conll results
    model.evaluate(session, model_id = sys.argv[1], official_stdout=True, eval_mode=True)
