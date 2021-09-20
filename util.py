from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import shutil
import sys
from csv import DictWriter

import numpy as np
import tensorflow as tf
import pyhocon

import models.independent
import models.overlap
import models.independent_az_spans
import models.independent_projection
import models.indep_knowledge
import models.independent_emb
import models.independent_concept_clutsers
import models.non_context_model
import models.joint_loss
from debug_util import load_pickle
from emb_master import SAMPLE_DIR
import json
import pandas as pd
import yaml



def custom_distance_coref(cluster_ids):
  coref_distances = tf.constant(1.0) - tf.to_float(
    tf.equal(tf.subtract(tf.expand_dims(cluster_ids, 1), cluster_ids), tf.constant(0)))
  inv_ids = tf.expand_dims(tf.to_float(tf.equal(cluster_ids, tf.constant(-1))), 1)
  inv_mask = tf.not_equal(tf.matmul(inv_ids, tf.transpose(inv_ids)), tf.constant(1.0))

  valid_coref_distances = tf.where(inv_mask, coref_distances, tf.ones_like(inv_mask, dtype=float))
  diag = tf.to_float(tf.zeros_like(cluster_ids))
  valid_coref_distances = tf.linalg.set_diag(valid_coref_distances, diag)
  return valid_coref_distances

def custom_distance_concepts(concept_ids):
  concept_distances = tf.constant(1.0) - tf.to_float(
    tf.equal(tf.subtract(tf.expand_dims(concept_ids, 1), concept_ids), tf.constant(0)))
  inv_ids = tf.expand_dims(tf.to_float(tf.logical_or(
    tf.equal(concept_ids, tf.constant(0)),
    tf.equal(concept_ids, tf.constant(5)))), 1)
  inv_mask = tf.not_equal(tf.matmul(inv_ids, tf.transpose(inv_ids)), tf.constant(1.0))

  valid_concept_distances = tf.where(inv_mask, concept_distances, tf.ones_like(inv_mask, dtype=float))
  diag = tf.to_float(tf.zeros_like(concept_ids))
  valid_concept_distances = tf.linalg.set_diag(valid_concept_distances, diag)
  return valid_concept_distances

def tf_cosine_distance(x):
  x = tf.to_float(x)
  similarity = tf.reduce_sum(x[:, tf.newaxis] * x, axis=-1)
  # Only necessary if vectors are not normalized
  similarity /= tf.norm(x[:, tf.newaxis], axis=-1) * tf.norm(x, axis=-1)
  # If you prefer the distance measure
  distance = 1 - similarity
  return distance

def get_model(config, args=None):
    if config['model_type'] == 'independent':
        return models.independent.CorefModel(config)
    elif config['model_type'] == 'overlap':
        return models.overlap.CorefModel(config)
    elif config['model_type'] == 'independent_az_spans':
        return models.independent_az_spans.CorefModel(config)
    elif config['model_type'] == 'independent_projection':
        return models.independent_projection.CorefModel(config)
    elif config['model_type'] == 'independent_knowledge':
        return models.indep_knowledge.CorefModel(config)
    elif config['model_type'] == 'projection':
        return models.independent_emb.ProjectionModel(config, args)
    elif config['model_type'] == 'concept_cl':
      return models.independent_concept_clutsers.CorefConceptModel(config)
    elif config['model_type'] == 'noncontext':
      return models.non_context_model.NonContextModel(config, args)
    elif config['model_type'] == 'joint':
      return models.joint_loss.JointLossModel(config, args)
    else:
        raise NotImplementedError('Undefined model type')

def initialize_from_env():
  if "GPU" in os.environ:
    print('[initialize_from_env]', 'GP in os.environ')

    set_gpus(int(os.environ["GPU"]))
  else:
    print('[initialize_from_env]GP not in os.environ')

  name = sys.argv[1]
  config_file = sys.argv[2]
  print("Running experiment: {}".format(name))

  # if eval_test:
  #   config = pyhocon.ConfigFactory.parse_file("test.experiments.conf")[name]
  # else:
  #   config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  # print(pyhocon.ConfigFactory.parse_file(config_file))
  config  = pyhocon.ConfigFactory.parse_file(config_file)[name]

  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config

def initialize_yaml_from_env(name):
  print("Running projection: {}".format(name))
  yaml_args= yaml.load(open(name))
  return yaml_args




def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def save_results(conll_results, model_id, dataset, output_dir, filename, jsonlines_predicted):
  toCSV = {}
  for eval_scheme in conll_results.keys():
    for metric in conll_results[eval_scheme].keys():
      toCSV['{}_{}'.format(eval_scheme, metric)] = conll_results[eval_scheme][metric]
  toCSV['model_id'] = model_id
  toCSV['dataset'] = dataset
  with open(os.path.join(output_dir, filename), 'a') as f_object:

    # Pass the file object and a list
    # of column names to DictWriter()
    # You will get a object of DictWriter
    dictwriter_object = DictWriter(f_object, fieldnames=toCSV.keys())

    # Pass the dictionary as an argument to the Writerow()
    dictwriter_object.writerow(toCSV)

    # Close the file object
    f_object.close()

  count = 0
  with open(os.path.join(output_dir, model_id+'_predicted.jsonlines'), "w") as output_file:

    for document in jsonlines_predicted:

      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1

  print("Wrote {} documents to {}".format(count, os.path.join(output_dir, model_id+'_predicted.jsonlines')))





def flatten(l):
  return [item for sublist in l for item in sublist]

def flatten_concepts(l):
  ret = []
  for k in ['person', 'treatment','test', 'problem','pronoun']:
    if k in l:
      ret.append(l[k])
    else:
      ret.append([])
  return ret

def flatten_umls(l, umls_csv):
  ret = []
  # load concepts
  concept_csv = pd.read_csv(umls_csv, names=['cui', 'category_name'], usecols=[1, 2]).iloc[1: , :]
  cui_labels = concept_csv['cui'].tolist()
  for k in cui_labels:
    if k in l:
      ret.append(l[k])
    else:
      ret.append([])
  return ret



def set_gpus(*gpus):
  # pass
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
  print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=tf.truncated_normal_initializer(stddev=0.02)):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

def highway(inputs, num_layers, dropout):
  for i in range(num_layers):
    with tf.variable_scope("highway_{}".format(i)):
      j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
      f = tf.sigmoid(f)
      j = tf.nn.relu(j)
      if dropout is not None:
        j = tf.nn.dropout(j, dropout)
      inputs = f * j + (1 - f) * inputs
  return inputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02), hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size], initializer=hidden_initializer)
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
  return outputs

def linear(inputs, output_size):
  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs
  hidden_weights = tf.get_variable("projection_linear_w", [shape(current_inputs, 1), output_size[1]], trainable=True)
  hidden_bias = tf.get_variable("projection_bias", [output_size[1]], trainable=True)
  current_outputs = tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias)
  return current_outputs

def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def batch_gather(emb, indices):
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
  if len(emb.get_shape()) == 2:
    gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
  return gathered

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings = self.load_embedding_dict(self._path)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.zeros(self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):
          word_end = line.find(" ")
          word = line[:word_end]
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      print("Done loading word embeddings.")
    return embedding_dict

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
