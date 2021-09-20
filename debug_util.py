import json
import h5py
import pickle
import os
from emb_master import SAMPLE_DIR
from conll import output_conll, official_conll_eval
import tempfile
import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.decomposition import PCA

def get_dataset(filename):
    params = filename.split('_')
    if 'dataset=src' in params:
        return 'src'
    elif 'dataset=tgt' in params:
        return 'tgt'
    else:
        print('[get_dataset] ERROR INVALID FILENAME (MISSING DATASET)', filename)
        exit(0)
def get_partition(filename):
    params = filename.split('_')
    if 'partition=train' in params:
        return 'train'
    elif 'partition=dev' in params:
        return 'dev'
    elif 'partition=test' in params:
        return 'test'
    else:
        print('[get_partition] ERROR INVALID FILENAME (MISSING PARTITION)', filename)
        exit(0)


def get_sample_jsonlines(sample_id, jsonlines_file):
    print('[get_sample_jsonlines] in function')
    with open(jsonlines_file) as f:
        jsonlines = [json.loads(jsonline) for jsonline in f.readlines()]
    print('[get_sample_jsonlines] loaded jsonlines', jsonlines_file)

    # load sample_id doc_ids
    all_doc_keys = load_pickle(os.path.join(SAMPLE_DIR, 'sample_dict.pickle'))
    if 'i2b2' in jsonlines_file:
        dataset = 'tgt'
    else:
        dataset = 'src'
    if 'train' in jsonlines_file:
        partition = 'train'
    elif 'dev' in jsonlines_file:
        partition = 'dev'
    else:
        print('[get_sample_jsonlines] ERROR INVALID FILENAME (MISSING PARTITION)', jsonlines_file)
        return jsonlines
        # exit(0)
    # print('all_doc_keys keys', all_doc_keys.keys())
    doc_keys = all_doc_keys[sample_id]['{}_{}_keys'.format(dataset, partition)]
    print('[get_sample_jsonlines] got doc_keys', len(doc_keys), 'partition', partition, doc_keys)

    sample = []
    for x in jsonlines:
        if x['doc_key'] in doc_keys:
            sample.append(x)
    print('[get_sample_jsonlines] sample_len', len(sample))
    return sample



def get_sample_hdf5(sample_id, filepath):
  # load sample_id doc_ids
  all_doc_keys = load_pickle(os.path.join(SAMPLE_DIR, 'sample_dict.pickle'))
  doc_keys = all_doc_keys[sample_id]['{}_{}_keys'.format(get_dataset(filepath), get_partition(filepath))]
  print('[get_sample_hdf5] doc_keys', doc_keys)

  hf = h5py.File(filepath, 'r')
  print('filename', filepath)
  print('hf_keys:', hf.keys())

  sample_hf = {}
  for key in doc_keys:
    if str(key) not in hf.keys():
        print('error key not found in hf file:', key)
        exit(0)
    else:
        sample_hf[str(key)] = hf[str(key)]
  return sample_hf


# read the jsonlines file
def load_jsonlines(jsonfile_loc):
    with open(jsonfile_loc) as input_file:
        print('[load_jsonlines]', jsonfile_loc)
        num_lines = sum(1 for line in input_file.readlines())
        input_file.seek(0)  # return to first line
        # print('at beginning of file', len(input_file.readlines()))
        lines = input_file.readlines()
    return [json.loads(line) for line in lines]

def hdf5_indices(filepath):
    print('filepath:', filepath)
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    return indices

def save_pickle(content, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(content, fp, protocol=pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


def get_sample_ids(maxex, sample_unq_id):
  """given some max number of examples, we have every interval of 10"""
  sids = []
  for num_tgt in range(0, maxex+1, 10):
    num_src = maxex-num_tgt
    sid = '{}-{}-{}'.format(num_tgt, num_src, sample_unq_id)
    sids.append(sid)
  return sids

def flatten_concepts(l):
  ret = []
  for k in ['person', 'treatment','test', 'problem','pronoun']:
    if k in l:
      ret.append(l[k])
    else:
      ret.append([])
  return ret

def flatten_umls(l, umls_csv='/projects/tir4/users/nmgandhi/coref/data/umls/cui_merged.csv'):
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


def parse_sample_id(model_id):
  """parse out the sample_id from the model_id"""
  for param in model_id.split('_'):
      if param.startswith('sid'):
          sample_id = param[len('sid'):]
          return sample_id
  print('ERROR THERE IS NO SAMPLE ID HERE', model_id)
  return '10-90-0'
def get_conf_name(model, seg_len, bert_lr, task_lr, projection_params, sample_id='', tune=False, task_optimizer=None, eps=None):


    tune_name = '_tune{}'.format(tune)



    if projection_params is not None:
        projection_name = '_'+ format_projection_id(projection_params['sample_id'],
                                               projection_params['emb_weight'],
                                               projection_params['doc_level'],
                                               projection_params['max_rank'])
        projection_name += '_cat{}'.format(projection_params['concat'])
    else:
        projection_name = ""
        if tune:
            projection_name += '_sid{}'.format(sample_id)

    # print('[get_conf_name] projetioN_name', projection_name)
    if task_optimizer is None and eps is None:
        return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr) + projection_name + tune_name

    else:
        return '{}_sl{}_blr{}_tlr{}_to{}_eps{}'.format(model, seg_len, bert_lr, task_lr, task_optimizer, eps) + projection_name + tune_name

def get_conf_name_abl(model, seg_len, bert_lr, task_lr, yaml_args, concat, tune=False, task_optimizer=None, eps=None):
    tune_name = '_tune{}'.format(tune)

    if yaml_args is not None:
        projection_name = '_'+ format_projection_id_abl(yaml_args)
        projection_name += '_cat{}'.format(concat)
    else:
        print('ERROR PROJECTION PARAMS MISSING')
        exit(0)
    if task_optimizer is None and eps is None:
        return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr) + projection_name + tune_name

    else:
        return '{}_sl{}_blr{}_tlr{}_to{}_eps{}'.format(model, seg_len, bert_lr, task_lr, task_optimizer, eps) + projection_name + tune_name



    pass

def format_projection_id_abl(yaml_args):
    base_projection_name = 'sid{}_dl{}_mr{}_cw{}_kw{}_sw{}'.format(yaml_args['dataset']['sample_id'],
                                                    yaml_args['dataset']['doc_level'],
                                                    yaml_args['probe']['maximum_rank'],
                                                    yaml_args['dataset']['coref_weight'],
                                                    yaml_args['dataset']['concept_weight'],
                                                    yaml_args['dataset']['emb_weight'])
    return base_projection_name


def get_conf_name_pjl(model, seg_len, bert_lr, task_lr, yaml_args, concat, plw, tune=False, task_optimizer=None, eps=None):
    tune_name = '_tune{}_plw{}'.format(tune, plw)

    if yaml_args is not None:
        projection_name = '_'+ format_projection_id_pjl(yaml_args, tune)
        projection_name += '_cat{}'.format(concat)
    else:
        print('ERROR PROJECTION PARAMS MISSING')
        exit(0)
    if task_optimizer is None and eps is None:
        return '{}_sl{}_blr{}_tlr{}'.format(model, seg_len, bert_lr, task_lr) + projection_name + tune_name

    else:
        return '{}_sl{}_blr{}_tlr{}_to{}_eps{}'.format(model, seg_len, bert_lr, task_lr, task_optimizer, eps) + projection_name + tune_name



    pass

def format_projection_id_pjl(yaml_args, tune):
    if tune:
        base_projection_name = 'sid{}_dl{}_mr{}_cw{}_kw{}_sw{}_uw{}'.format(yaml_args['dataset']['sample_id'],
                                                    yaml_args['dataset']['doc_level'],
                                                    yaml_args['probe']['maximum_rank'],
                                                    yaml_args['dataset']['coref_weight'],
                                                    yaml_args['dataset']['concept_weight'],
                                                    yaml_args['dataset']['emb_weight'],
                                                    yaml_args['dataset']['umls_weight'])
    else:
        base_projection_name = 'dl{}_mr{}_cw{}_kw{}_sw{}_uw{}'.format(yaml_args['dataset']['doc_level'],
                                                               yaml_args['probe']['maximum_rank'],
                                                               yaml_args['dataset']['coref_weight'],
                                                               yaml_args['dataset']['concept_weight'],
                                                               yaml_args['dataset']['emb_weight'],
                                                               yaml_args['dataset']['umls_weight'])
    return base_projection_name



def format_projection_id(sample_id, w, doc_level, max_rank, concept=False):

    if concept:
        return 'sid{}_w{}_dl{}_mr{}_c{}'.format(sample_id, w, doc_level, max_rank, concept)
    else:
        return 'sid{}_w{}_dl{}_mr{}'.format(sample_id, w, doc_level, max_rank)

def compute_projection(x,y):
    # compute the projection of x onto y
    return y * np.dot(x, y) / np.dot(y, y)



def update_span_projections(candidate_starts, candidate_ends, candidate_span_emb,
                            span_pairs, doc_key):
    # for each span pair (x,y)
    # find the specific span_embs
    # compute the projection matrix
    # save the projections


    projections = []
    for span_pair in span_pairs:
        x_indices = tuple(span_pair['x'])
        y_indices = tuple(span_pair['y'])
        cand_indices = list(zip(candidate_starts, candidate_ends))
        assert x_indices in list(cand_indices), 'x indices not in cand indicies'
        assert y_indices in list(cand_indices), 'y indices not in cand indicies'
        cand_x_idx = cand_indices.index(x_indices)
        cand_y_idx = cand_indices.index(y_indices)
        print('[update_span_projections] candidate_span_emb[cand_x_idx]', candidate_span_emb[cand_x_idx].shape)
        print('[update_span_projections] candidate_span_emb[cand_y_idx]', candidate_span_emb[cand_y_idx].shape)

        projection_matrix = compute_projection(np.transpose(candidate_span_emb[cand_x_idx]),
                                               np.transpose(candidate_span_emb[cand_y_idx]))
        print('[update_span_projections]projection_matrix', projection_matrix.shape)

        projections.append({'proj_name':'x:{}_y:{}_dockey:{}'.format(span_pair['x_text'],span_pair['y_text'], doc_key),
                           'projection':projection_matrix,
                            'x_emb':candidate_span_emb[cand_x_idx]})
    return projections





def update_span_distances(candidate_starts, candidate_ends, candidate_span_emb,
                   subtokens_mrg, text_of_interest, span_pairs, distances):
    span_emb_of_interest = {}
    for st, end, emb in zip(candidate_starts, candidate_ends, candidate_span_emb):
        span_text = reformat_str(subtokens_mrg[st:end + 1])
        if span_text.strip().lower() in text_of_interest:
            # print('found span_text', span_text)
            span_emb_of_interest[span_text] = emb

    for sp_text1, sp_text2 in span_pairs:
        if sp_text1 in span_emb_of_interest and sp_text2 in span_emb_of_interest:
            distance = 1.0- spatial.distance.cosine(span_emb_of_interest[sp_text1].tolist(),
                                               span_emb_of_interest[sp_text2].tolist())
            distances['({},{})'.format(sp_text1, sp_text2)].append(distance)
            # print('found a pair of incompat ({}, {}) with cosine sim {}'.format(sp_text1, sp_text2,
            #                                                                     distance))

    return distances
def reformat_str(subtoken_list):
    subtoken_str = ''
    for i, x in enumerate(subtoken_list):
        if (x[0] == '#'):
            subtoken_str += x
        else:
            subtoken_str += (" " + x)
    return subtoken_str.replace('#', '').strip()


def count_subwords(subtoken_list):
    count = 0
    for x in subtoken_list:
        if x.startswith('##'):
            count += 1
    return count *1.0


# true if there is a correctly identified span in predicted clusters overlaping
def search_overlapping(cluster, span_indices):
    for start_idx, end_idx in cluster:
        # there is a predicted span that is a subset of span_indices
        if start_idx >= min(span_indices) and end_idx <= max(span_indices):
            return True
        # there is a predicted span that is a superset of span_indices
        if start_idx <= min(span_indices) and end_idx >= max(span_indices):
            return True
    return False

""" Take a set of spans (texts), clusters   
    output all the clusters containing at least one span from that set of spans
"""
def get_clusters_containing_indices(span_set, clusters):
    clusters_containing = []
    for cl in clusters:
        for span_indices in span_set:
            if search_overlapping(cl, span_indices):
                clusters_containing.append(cl)
                break
    return clusters_containing


""" Take a set of spans (texts), clusters   
    output all the clusters containing at least one span from that set of spans
"""
def get_clusters_containing(span_set, clusters, subtokens_mrg):
    clusters_containing = []
    for cl in clusters:
        for start_idx, end_idx in cl:
            if reformat_str(subtokens_mrg[start_idx:end_idx+1]) in span_set:
                clusters_containing.append(cl)
                break
    return clusters_containing

""" Take a set of spans (texts), clusters   
    output all the clusters with average subtoken size within binsize
"""
def get_clusters_subtoken_range(clusters, subtokens_mrg, bin_range):
    clusters_range = []
    for cl in clusters:
        cl_subword_ave = 0.0
        for start_idx, end_idx in cl:
            cl_subword_ave += count_subwords(subtokens_mrg[start_idx:end_idx + 1])
        cl_subword_ave /= len(cl)
        # print('[get_clusters_subtoken_range] cl_subword_ave', cl_subword_ave)
        if cl_subword_ave >=bin_range[0] and cl_subword_ave <= bin_range[1]:
            clusters_range.append(cl)
    return clusters_range



""" Take a set of clusters and subtokens_mrg for one doc
    output the average number of subtokens per span in each cluster
"""
def get_subtoken_cl_ave(clusters, subtokens_mrg):
    cluster_averages = []
    for cl in clusters:
        cl_subword_ave = 0.0
        for start_idx, end_idx in cl:
            cl_subword_ave += count_subwords(subtokens_mrg[start_idx:end_idx+1])
        cl_subword_ave /= len(cl)
        cluster_averages.append(cl_subword_ave)
    return cluster_averages





# Get the spans in the training set that occurs at most N times
# Get the clusters in the gold set of clusters containing one of these spans
# Get the clusters in the predicted set of clusters containing one of these spans

""" Get the set of spans occurring at most N times 
    given, a span train frq dict, clusters
"""
def get_frequent_spans(train_frq_dict, N):

    span_texts = []
    for span, frq in train_frq_dict.itemize():
        if frq < N:
            span_texts.append(span)

    return span_texts


""" Take a list of (doc_id, 
                    cluster_id, 
                    predicted clusters including ONLY spans from cluster id pairs,
                    actual cluster) tuples
    predictions should be a dict of doc_key:clusters
    Gold conll file
    subtoken_map
    produce two conll files (use the output_conll function x2) and then evaluate
    return f-score { m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }

    
"""
def evaluate_cluster_subset(gold_path, subtoken_maps, predictions, gold_clusters):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", prefix='/projects/tir4/users/nmgandhi/coref/debug/tmp_files/') as prediction_subset_file:
      with open(gold_path, "r") as gold_file:
        # print('writing', predictions, subtoken_maps)
        output_conll(gold_file, prediction_subset_file, predictions, subtoken_maps)
    with tempfile.NamedTemporaryFile(delete=False, mode="w", prefix='/projects/tir4/users/nmgandhi/coref/debug/tmp_files/') as gold_subset_file:
      with open(gold_path, "r") as gold_file:
            # print('gold_clusters', gold_clusters)
        output_conll(gold_file, gold_subset_file, gold_clusters, subtoken_maps)
    print('written to gold_subset_file', gold_subset_file.name)
    print('written to prediction file', prediction_subset_file.name)


    results = { m: official_conll_eval(gold_subset_file.name, prediction_subset_file.name, m, False) for m in ("muc", "bcub", "ceafe") }

    os.remove(gold_subset_file.name)
    os.remove(prediction_subset_file.name)
    return results


    pass