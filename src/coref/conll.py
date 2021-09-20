from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tempfile
import subprocess
import operator
import os
import collections

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
BEGIN_DOCUMENT_REGEX_I2B2 = re.compile(r"#begin document (.*)")
BEGIN_DOCUMENT_REGEX_I2B2_ALT = re.compile(r"#begin document (.*);")

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def get_doc_key(doc_id, part):
  return "{}_{}".format(doc_id, int(part))

def output_conll(input_file, output_file, predictions, subtoken_map):
  prediction_map = {}
  # print('[output_conlll] predictions', predictions)
  for doc_key, clusters in predictions.items():
    # print('[output_conll] clusters len', doc_key, len(clusters))
    start_map = collections.defaultdict(list)
    end_map = collections.defaultdict(list)
    word_map = collections.defaultdict(list)
    for cluster_id, mentions in enumerate(clusters):
      for start, end in mentions:
        start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
        if start == end:
          word_map[start].append(cluster_id)
        else:
          start_map[start].append((cluster_id, end))
          end_map[end].append((cluster_id, start))
    for k,v in start_map.items():
      start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k,v in end_map.items():
      end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
    prediction_map[doc_key] = (start_map, end_map, word_map)
  # print('[output_conll] prediction_map', str(prediction_map)[:200])


  word_index = 0
  print('input_file:', input_file)
  for line in input_file.readlines():
    # print('[output_conll] line', line)
    row = line.split()
    if len(row) == 0:
      output_file.write("\n")
    elif row[0].startswith("#"):
      # print('found begin', line)
      if('clinical' in line):
        if ';' in line:
          # print('recognized meta')
          begin_match = re.match(BEGIN_DOCUMENT_REGEX_I2B2_ALT, line)
        else:
          # print('did not recognize meta', input_file)
          begin_match = re.match(BEGIN_DOCUMENT_REGEX_I2B2, line)
      else:
        begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
      # print('bgein_match', begin_match)

      if begin_match:
        if ('clinical' in line):
          doc_key = get_doc_key(begin_match.group(1), 0)
        else:
          doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
        # print('[output_conll]doc_key', doc_key)
        # doc_key = get_doc_key(begin_match.group(1), 0)
        # print('[output_conll]doc_key', doc_key)

        start_map, end_map, word_map = prediction_map[doc_key]
        word_index = 0
      output_file.write(line)

      output_file.write("\n")
    else:
      # assert get_doc_key(row[0], row[1]) == doc_key
      coref_list = []
      if word_index in end_map:
        for cluster_id in end_map[word_index]:
          coref_list.append("{})".format(cluster_id))
      if word_index in word_map:
        for cluster_id in word_map[word_index]:
          coref_list.append("({})".format(cluster_id))
      if word_index in start_map:
        for cluster_id in start_map[word_index]:
          coref_list.append("({}".format(cluster_id))

      if len(coref_list) == 0:
        row[-1] = "-"
      else:
        row[-1] = "|".join(coref_list)

      output_file.write("   ".join(row))

      # exit(0)
      output_file.write("\n")
      word_index += 1

def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False):
  cmd = ["/projects/tir4/users/nmgandhi/coref/src/coref/conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]

  # cmd = ["conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  stdout, stderr = process.communicate()
  process.wait()

  stdout = stdout.decode("utf-8")
  if stderr is not None:
    print(stderr)

  if official_stdout:
    print("Official result for {}".format(metric))
    print(stdout)

  coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
  recall = float(coref_results_match.group(1))
  precision = float(coref_results_match.group(2))
  f1 = float(coref_results_match.group(3))
  return { "r": recall, "p": precision, "f": f1 }

def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=False, output_conll_file=''):
  print('[evaluate_conll] entered function')
  if len(output_conll_file) > 0:
    with open(output_conll_file, 'w') as prediction_file:
      with open(gold_path, "r") as gold_file:
        output_conll(gold_file, prediction_file, predictions, subtoken_maps)
      print("Predicted conll file: {}".format(prediction_file.name))
  else:
    with tempfile.NamedTemporaryFile(delete=False, mode="w", prefix='/projects/tir4/users/nmgandhi/coref/debug/') as prediction_file:
      with open(gold_path, "r") as gold_file:
        output_conll(gold_file, prediction_file, predictions, subtoken_maps)
    print("Predicted conll file: {}".format(prediction_file.name))
  print('[evaluate_conll] returning from function')
  results =  { m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
  # os.remove(prediction_file.name)
  return results