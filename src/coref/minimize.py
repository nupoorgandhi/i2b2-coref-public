from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import tempfile
import subprocess
import collections
from string import punctuation
import util
import conll
from bert import tokenization
import shlex
from collections import defaultdict
import copy
import itertools
import debug_util
import glob
import json

CONCEPT_CODE_DICT = {'problem':1,
                     'person':2,
                     'test':3,
                     'treatment':4,
                     'pronoun':5}

class DocumentState(object):
  def __init__(self, key, concept_dict_str, umls_dict_str):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.clusters = collections.defaultdict(list)
    self.concept_clusters = {}
    self.coref_stacks = collections.defaultdict(list)
    self.concept_stacks = collections.defaultdict(list)

    self.speakers = []
    self.segment_info = []
    self.token_ids =[]
    
    self.concept_dict_str = concept_dict_str
    self.umls_dict_str = umls_dict_str
    self.umls_clusters = {}

  def get_concept_dict(self):
    concept_dict = {}
    directory = ''
    for filename in os.listdir(directory):
      doc_id = filename[:-len('.txt.con')]
      concept_dict[doc_id] = defaultdict(list)
      with open(os.path.join(directory, filename)) as fp:
        for line in fp.readlines():
          
          try:
            shlex_split_str = shlex.split(line.strip())
          except (ValueError) as e:
            print('caught a value error for ', line.strip())
            continue


          if len(shlex_split_str[1].split(':')) > 1:
            concept_sen_id, concept_start = shlex.split(line.strip())[1].split(':')
            concept_sen_id = int(concept_sen_id)
            concept_start = int(concept_start)
            concept_sen_id_2, concept_end = shlex.split(line.strip())[2].split('||')[0].split(':')
            concept_sen_id_2 = int(concept_sen_id_2)
            concept_end = int(concept_end)

            if concept_sen_id_2 != concept_sen_id:
              print(doc_id, 'weird non-aligned concept_sen_id, concept_sen_id_2', concept_sen_id, concept_sen_id_2, doc_id)
            else:
              concept = shlex.split(line.strip())[2].split('||')[1]
              concept = concept.split('=')[1]
              concept_dict[doc_id][concept_sen_id].append((concept_start, concept_end,concept))
    print('set up concept dict, length', len(concept_dict.keys()))
    return concept_dict



  def retrieve_concept(self, doc_id, token_id):
    return 0
    pprefix = '[retrieve_concept]'
    doc_id = doc_id.split('_')[0]
    if doc_id not in self.concept_dict:
      return 0
    sen_id = int(token_id.split(':')[0])
    if sen_id not in self.concept_dict[doc_id]:
      return 0
    if len(token_id.split(':')) ==1:
      return 0
    else:
      token_id = int(token_id.split(':')[1])
    for (concept_start, concept_end, concept) in self.concept_dict[doc_id][sen_id]:
      if token_id <= concept_end and token_id >= concept_start:
        return CONCEPT_CODE_DICT[concept.strip().lower()]
    return 0

  def retrieve_str_subtoken_map(self, subtoken_map):
    start_text = start_text.lower().strip()#.strip(punctuation)
    end_text = end_text.lower().strip()#.strip(punctuation)

    for (st, end, concept, text) in self.concept_dict_str[doc_id]:
      if st == start_text and end == end_text:
        return CONCEPT_CODE_DICT[concept]
    print('FAILED TO RETRIEVE')
    return 0



  def finalize(self):
    pprefix = '[finalize]'
    
    subtoken_idx = 0

    for segment in self.segment_info:
      speakers = []
      token_ids = []

      for i, tok_info in enumerate(segment):
        speakers.append('[SPL]')
        if tok_info is not None:

          concept = self.retrieve_concept(self.doc_key, tok_info[4])
          
        else:
          concept = 0
        token_ids.append(concept)
        
      self.speakers += [speakers]
      self.token_ids += [token_ids]
    # populate sentence map

    # i2b2_subtoken_id_map = []
    subtoken_text = []
    # populate clusters
    first_subtoken_index = -1
    for seg_idx, segment in enumerate(self.segment_info):
      speakers = []

      for i, tok_info in enumerate(segment):
        first_subtoken_index += 1
        coref = tok_info[-2] if tok_info is not None else '-'
        i2b2_id = tok_info[-3] if tok_info is not None else '-'
        conll_text = tok_info[1] if tok_info is not None else '-'
        
        if coref != "-":

          last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
          for part in coref.split("|"):
            if part[0] == "(":
              if part[-1] == ")":
                cluster_id = int(part[1:-1])

                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
              else:
                cluster_id = int(part[1:])
                self.coref_stacks[cluster_id].append(first_subtoken_index)
            else:
              cluster_id = int(part[:-1])
              start = self.coref_stacks[cluster_id].pop()
              self.clusters[cluster_id].append((start, last_subtoken_index))


    subtoken_text = list(itertools.chain.from_iterable(self.segments))

    concept_clusters_text = defaultdict(list)
    self.concept_clusters[self.doc_key] = defaultdict(list)
    self.umls_clusters[self.doc_key] = defaultdict(list)
    if 'clinical' in self.doc_key:
      for i in range(len(subtoken_text)):
        # conll_text = copy.deepcopy(conll_text_i)
        for j in range(100):


          conll_text = debug_util.reformat_str(subtoken_text[i:j+i+1]).lower().strip().replace(" ", "")#.strip(punctuation)

          if self.doc_key in self.concept_dict_str and conll_text in self.concept_dict_str[self.doc_key]:# and j-i <= len(conll_text.split()):
            concept = self.concept_dict_str[self.doc_key][conll_text]
            self.concept_clusters[self.doc_key][concept].append([i, j+i])
            concept_clusters_text[concept].append(conll_text)
          if self.doc_key in self.umls_dict_str and conll_text in self.umls_dict_str[self.doc_key]:
            umls_cid = self.umls_dict_str[self.doc_key][conll_text]
            self.umls_clusters[self.doc_key][umls_cid].append([i, j + i])




   
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]

  
    all_mentions = util.flatten(merged_clusters)
    sentence_map =  get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = util.flatten(self.segment_subtoken_map)
    assert len(all_mentions) == len(set(all_mentions))
    num_words =  len(util.flatten(self.segments))
    assert num_words == len(util.flatten(self.speakers))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      "doc_key": self.doc_key,
      "sentences": self.segments,
      "speakers": self.speakers,
      "constituents": [],
      "ner": [],
      "clusters": merged_clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
      'pronouns': self.pronouns,
      'token_ids':self.token_ids,
      "concept_clusters":self.concept_clusters[self.doc_key],
      "umls_clusters":self.umls_clusters[self.doc_key]
    }


def normalize_word(word, language):
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
  current = 0
  previous_token = 0
  while current < len(document_state.subtokens):
    end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current : end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current : end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map

def get_document_onto(document_lines, tokenizer, language, segment_len):
  document_state = DocumentState(document_lines[0],{}, {})
  word_idx = -1
  for line in document_lines[1]:
    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      # assert len(row) >= 12
      # assert len(row) >= 6
      word_idx += 1
      #TODO
      word = normalize_word(row[3], language)

      # word = normalize_word(row[0], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  # split_into_segments(document_state, segment_len, document_state.token_end)
  # split_into_segments(document_state, segment_len, document_state.sentence_end)
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document

def get_document_i2b2(document_lines, tokenizer, language, segment_len, concept_dict_str, umls_dict_str):
  document_state = DocumentState(document_lines[0], concept_dict_str, umls_dict_str)
  word_idx = -1
  for line in document_lines[1]:
    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      # assert len(row) >= 12
      # assert len(row) >= 6
      word_idx += 1
      #TODO
      # word = normalize_word(row[3], language)
      if language == 'meta':
        word = normalize_word(row[1], language)
      else:
        word = normalize_word(row[0], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  # split_into_segments(document_state, segment_len, document_state.token_end)
  # split_into_segments(document_state, segment_len, document_state.sentence_end)
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document

def skip(doc_key):
  # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
  return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
  input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
  output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document_onto(document_lines, tokenizer, language, seg_len)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))


def get_concept_dict():
  concept_dict = {}
  directory = '/projects/tir4/users/nmgandhi/coref/data/concepts'
  # iterate through docs in directory
  for filename in os.listdir(directory):
    doc_id = filename[:-len('.txt.con')]
    concept_dict[doc_id] = defaultdict(list)
    with open(os.path.join(directory, filename)) as fp:
      for line in fp.readlines():

        try:
          shlex_split_str = shlex.split(line.strip())
        except (ValueError) as e:
          print('caught a value error for ', line.strip())
          continue


        if len(shlex_split_str[1].split(':')) > 1:
          concept_sen_id, concept_start = shlex.split(line.strip())[1].split(':')
          concept_sen_id = int(concept_sen_id)
          concept_start = int(concept_start)
          concept_sen_id_2, concept_end = shlex.split(line.strip())[2].split('||')[0].split(':')
          concept_sen_id_2 = int(concept_sen_id_2)
          concept_end = int(concept_end)

          if concept_sen_id_2 != concept_sen_id:
            print(doc_id, 'weird non-aligned concept_sen_id, concept_sen_id_2', concept_sen_id, concept_sen_id_2, doc_id)
          else:
            concept = shlex.split(line.strip())[2].split('||')[1]
            concept = concept.split('=')[1]
            concept_dict[doc_id][concept_sen_id].append((concept_start, concept_end,concept))
  print('set up concept dict, length', len(concept_dict.keys()))
  return concept_dict

def get_concept_dict_str(directory):
  concept_dict = {}
  # iterate through docs in directory
  for filename in os.listdir(directory):
    doc_id = filename[:-len('.txt.con')] + '_0'
    concept_dict[doc_id] = {}
    with open(os.path.join(directory, filename)) as fp:
      for line in fp.readlines():

        try:
          shlex_split_str = shlex.split(line.strip())
        except (ValueError) as e:
          print('caught a value error for ', line.strip())
          continue

        if len(shlex_split_str[1].split(':')) > 1:
          pieces = line.strip().split('||')

          i2b2_id_1 = shlex.split(line.strip())[1]
          # concept_sen_id = int(concept_sen_id)
          # concept_start = int(concept_start)
          i2b2_id_2 = shlex.split(line.strip())[2].split('||')[0]

          concept = shlex.split(line.strip())[2].split('||')[1]
          concept = concept.split('=')[1]

          span_text = shlex.split(line.strip())[0].split('=')[1].lower().strip().replace(" ", "")#.strip(punctuation)
          span_text_start = span_text.split()[0].lower().strip().replace(" ", "")#.strip(punctuation)
          span_text_end = span_text.split()[-1].lower().strip().replace(" ", "")#.strip(punctuation)

          concept_dict[doc_id][span_text] = concept#.append((span_text_start, span_text_end, concept, span_text))
  print('set up concept dict, length', len(concept_dict.keys()))
  return concept_dict
  
def get_umls_dict_str(directory):
  pprefix = '[get_umls_dict_str]'
  concept_dict = {}
  docs = glob.glob(directory + '/*.json', recursive=True)
  for doc in docs:
    doc_id = os.path.basename(doc)[:-len('.umls.json')] + '_0'
    concept_dict[doc_id] = {}

    with open(doc, 'r') as f:
      jsonlines = json.load(f)

    for sen in jsonlines:
      for ent in sen:
        ent_text = ent['pref_name'].lower().strip().replace(" ", "")
        concept_dict[doc_id][ent_text] = ent['cui']

  return concept_dict

def minimize_partition_i2b2(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir,
                            umls_dir, concept_dir):
  input_path = "{}/{}.{}".format(input_dir, name, extension)
  output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX_I2B2_ALT, line)
      #TODO
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), 0)
        # print('doc_key', doc_key)
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)
  umls_dict_str = get_umls_dict_str(umls_dir)

  concept_dict_str = get_concept_dict_str(concept_dir)

  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document_i2b2(document_lines, tokenizer, language, seg_len, concept_dict_str, umls_dict_str)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))





def minimize_language(language, labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case,
                      umls_dir, concept_dir):
  # do_lower_case = True if 'chinese' in vocab_file else False
  tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)

  minimize_partition_i2b2('test.i2b2', 'meta', 'meta.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, umls_dir, concept_dir)
  minimize_partition_i2b2('dev.i2b2', 'meta', 'meta.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, umls_dir, concept_dir)
  minimize_partition_i2b2('train.i2b2', 'meta', 'meta.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, umls_dir, concept_dir)

  # minimize_partition_i2b2('test.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)
  # minimize_partition_i2b2('dev.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)
  # minimize_partition_i2b2('train.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)

  minimize_partition("dev", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("train", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)



if __name__ == "__main__":

  # import os

  try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
  except KeyError:
    user_paths = []

  vocab_file = sys.argv[1]
  input_dir = sys.argv[2]
  output_dir = sys.argv[3]
  do_lower_case = sys.argv[4].lower() == 'true'
  umls_dir = sys.argv[5]
  concept_dir = sys.argv[6]
  print(do_lower_case)
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  for seg_len in [512, 256, 384, 128]:#[128, 256, 384, 512]:
    minimize_language("english", labels, stats, vocab_file, seg_len,
                      input_dir, output_dir, do_lower_case,
                      umls_dir, concept_dir)
    # minimize_language("chinese", labels, stats, vocab_file, seg_len)
    # minimize_language("es", labels, stats, vocab_file, seg_len)
    # minimize_language("arabic", labels, stats, vocab_file, seg_len)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
