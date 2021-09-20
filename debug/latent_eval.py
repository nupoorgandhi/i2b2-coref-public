import sys
import yaml
from argparse import ArgumentParser
import span_util
import h5py
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import glob
import csv
import json
import itertools
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from debug_util import load_jsonlines
"""

to evaluate target data: 
python latent_eval.py --project 1 --proj_root /projects/tir4/users/nmgandhi/coref/data/projections --span_emb_loc /projects/tir4/users/nmgandhi/coref/data/emb/model=spanbert_base_partition=test_dataset=tgt_maxex=10000.h5
to evaluate source data: 
python latent_eval.py --project 1 --proj_root /projects/tir4/users/nmgandhi/coref/data/projections --span_emb_loc /projects/tir4/users/nmgandhi/coref/data/emb/model\=spanbert_base_partition\=test_dataset\=src_maxex\=1000_ratio\=0.h5 
"""


def reformat_str(subtoken_list):
  subtoken_str = ''
  for i, x in enumerate(subtoken_list):
    if (x[0] == '#'):
      subtoken_str += x
    else:
      subtoken_str += (" " + x)
  return subtoken_str.replace('#', '').strip()


def generate_span_token_embeddings_from_hdf5(filepath):
    print('filepath:', filepath)
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    doc_embeddings_list = []
    start_indices_list = []
    end_indices_list = []
    cluster_ids_list = []
    doc_ids_list = []
    for idx in sorted([x for x in indices]):
    # for idx in indices:
        # print('idx:', idx)
        doc_embeddings = hf[str(idx)][0][:, :-3]
        start_indices = hf[str(idx)][0][:, -3]
        end_indices = hf[str(idx)][0][:, -2]
        cluster_ids = hf[str(idx)][0][:, -1]

        doc_embeddings_list.append(doc_embeddings)
        start_indices_list.append(start_indices)
        end_indices_list.append(end_indices)
        cluster_ids_list.append(cluster_ids)
        doc_ids_list.append(idx)
        # don't need to verify length since doc length has no relation with the number of spans
    return start_indices_list, end_indices_list, cluster_ids_list, doc_embeddings_list, doc_ids_list



# take as input the YAML file which contains the .h5 files you saved, projection
# apply projection (span_util)
# compute distances

def recover_span_repr(args, span_emb_loc, projection_path=None, to_project=False):
    """
    case1: span representation is the spanBERT span representation
        return from .h5 file
    case2: span representation is the spanBERT representation + projected into latent space
        load spans from .h5 file
        load projection
        apply projection
    case3: something else
        no worries for now

    :return: the span representations, coref integer labels
    """
    pprefix = '[recover_span_repr]'
    # span_emb_loc = os.path.join(args['dataset']['embeddings']['root'], args['dataset']['embeddings']['test_path'])

    print(pprefix, 'projection_path', projection_path)
    # recover spanBERT embeddings
    start_indices_list, end_indices_list, cluster_ids_list, doc_embeddings_list, doc_ids_list = generate_span_token_embeddings_from_hdf5(span_emb_loc)
    print(pprefix, 'len doc_embeddings_list', len(doc_embeddings_list))
    if not to_project:
        return start_indices_list, end_indices_list, cluster_ids_list, doc_embeddings_list, doc_ids_list

    else:
        B = span_util.recover_projection(projection_path) # [emb, rank]
        doc_embeddings_list_proj = []
        # span_emb = np.concatenate(doc_embeddings_list, axis=0)
        for span_emb in doc_embeddings_list:
            B = B.astype(np.float64)
            span_emb = span_emb.astype(np.float64)
            span_emb_proj = span_util.project_span_torch(span_emb, B)
            doc_embeddings_list_proj.append(span_emb_proj)
        return start_indices_list, end_indices_list, cluster_ids_list, doc_embeddings_list_proj, doc_ids_list

    pass

def gather_values_excluding(dict, key_to_exclude):
    values = []
    for k, v in dict.items():
        if k == key_to_exclude:
            continue
        else:
            values.append(v)
    return np.concatenate(values, axis=0)


def tsne_plot(args, labels, tokens, cluster_ids, projection_id=""):
    # "Creates and TSNE model and plots it"
    # labels = []
    # tokens = []
    #
    # for word in model.wv.vocab:
    #     tokens.append(model[word])
    #     labels.append(word)


    # create sets of indices for each cluster
    # num_cl = max(cluster_ids)
    # for i in range(int(num_cl)):
    #     indices = [i for i, x in enumerate(cluster_ids) if int(x) == int(i)]
    #     cluster_indices.append(indices)




    # print('label length:', len(labels), labels[:20])
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    # print('transofrimig tokens', tokens.shape)
    new_values = tsne_model.fit_transform(tokens)
    print('finished fita nd transofr tsne plot', new_values.shape)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])

        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # construct a cluster_id: indices dict
    cl_index_dict = defaultdict(list)
    for idx, cl_id in enumerate(cluster_ids):
        cl_index_dict[int(cl_id)].append(idx)


    for cl_id, cluster in cl_index_dict.items():
        if int(cl_id) >= 0:
            for span1_idx in cluster:
                for span2_idx in cluster:
                    plt.plot([x[span1_idx],x[span2_idx]],[y[span1_idx],y[span2_idx]], "green", linewidth=2, alpha=0.2)




    plt.savefig(os.path.join(args.outdir, 'latent_eval_tsne_plot_p={}_id={}.png'.format(args.project, projection_id)))
    print('saved to ', os.path.join(args.outdir, 'latent_eval_tsne_plot_p={}_id={}.png'.format(args.project, projection_id)))
def visualize_emb(args, doc_embeddings_list, cluster_ids_list, start_indices_list, end_indices_list, jsonlines_list, doc_ids_list, projection_id=""):
    """
    apply manifold dimensionality reduction to the span embeddings
    generate a plot for the span mebeddings and corresponding span text

    :param doc_embeddings_list:
    :param cluster_ids_list:
    :param start_indices_list:
    :param end_indices_list:
    :param jsonlines_list:
    :param doc_ids_list:
    :return:
    """
    pprefix = '[visualize_emb]'
    span_texts = []
    # unq_cluster_ids = defaultdict(list)
    unq_cluster_ids = []
    span_embs = np.concatenate(doc_embeddings_list,axis=0)

    # print(pprefix, 'doc_ids_list', doc_ids_list)
    # loop through docs
    cluster_id_count = 0
    idx = 0
    for span_emb, cluster_ids, start_ids, end_ids, doc_id in zip(doc_embeddings_list, cluster_ids_list,
                                                                 start_indices_list, end_indices_list, doc_ids_list):

        subtokens_mrg = []
        # print(pprefix, 'doc_id', doc_id)
        for x in jsonlines_list:
            # print(pprefix, 'x doc_key',  x['doc_key'])
            if doc_id == x['doc_key']:
                subtokens_mrg = list(itertools.chain.from_iterable(x['sentences']))

        for start_id, end_id in zip(start_ids, end_ids):
            # print(pprefix, 'max start', max(start_ids), 'max end', max(end_ids), 'len', len(subtokens_mrg))
            span_text = reformat_str(subtokens_mrg[int(start_id): int(end_id) + 1])
            # print(pprefix, ' what are we appending', span_text)
            span_texts.append(span_text)
            # exit(0)
        # if len(start_ids) != len(cluster_ids):
        #     print(pprefix, 'len(start_ids)', len(start_ids), 'len(cl_ids)', len(cluster_ids))
        cluster_id_count_local = 0
        for cl_id in cluster_ids:
            if cl_id == -1:
                # unq_cluster_ids[cl_id].append(idx)
                unq_cluster_ids.append(cl_id)

            else:
                unq_cluster_ids.append(cl_id + cluster_id_count)
                # unq_cluster_ids[cl_id + cluster_id_count].append(idx)
                cluster_id_count_local = max(cluster_id_count_local, cl_id)
            idx += 1
        # print('len(cluster_ids)', len(cluster_ids), 'idx', idx, 'len(span_texts)', len(span_texts))

        cluster_id_count += cluster_id_count_local
    # print(pprefix, 'final idx', idx)





    # print('[visualize_emb] span_texts', len(span_texts), span_texts[0])
    # exit(0)
    # print(pprefix, 'span_texts', span_texts[:-100])
    mask = np.random.choice([True, False], size=len(span_texts), p=[0.05, 0.95])
    numeric_spans = np.array([ (all(c.isalpha() for c in x)) for x in span_texts])
    mask &= numeric_spans

    # print(pprefix, 'numeric pasna', numeric_spans)
    # print(pprefix, 'masks', mask)
    # print(pprefix, 'before masking:np.array(span_texts)', np.array(span_texts).shape, 'span_embs', span_embs.shape)
    span_texts = np.array(span_texts)[mask]
    span_embs = span_embs[mask]
    unq_cluster_ids = np.array(unq_cluster_ids)[mask]
    # print(pprefix, 'after masking:np.array(span_texts)', np.array(span_texts).shape, 'span_embs', span_embs.shape)
    tsne_plot(args, span_texts, span_embs, unq_cluster_ids, projection_id)










    pass


def compute_distances(doc_embeddings_list, cluster_ids_list, start_indices_list, end_indices_list, jsonlines_list, doc_ids_list, n=5):
    """
    1. pairwise average distance between valid spans that corefer (exists per cluster)
    2. pairwise average distance between valid spans that do not corefer (exists per cluster)
    3. pairwise average distance between invalid span and valid span (exists per cluster and document)
    4. pairwise average distance between invalid spans
    :return:
    """
    # print('[compute_distances]jsonlines_list', jsonlines_list[0], type(jsonlines_list[0]))
    sentences_list = [x['sentences'] for x in jsonlines_list]

    pprefix = '[compute_distances]'
    cl_distance_sets = {'valid_coref':[], 'valid_no_coref':[], 'invalid_no_coref':[]}
    cl_distance_sets_text = {'valid_coref':[], 'valid_no_coref':[], 'invalid_no_coref':[]}

    doc_distance_sets = {'invalid_valid':[], 'invalid_invalid':[]}

    # loop through docs
    for span_emb, cluster_ids, start_ids, end_ids, doc_id in zip(doc_embeddings_list, cluster_ids_list,
                                                                    start_indices_list, end_indices_list, doc_ids_list):

        subtokens_mrg = []
        # print(pprefix, 'doc_id', doc_id)
        for x in jsonlines_list:
            # print(pprefix, 'x doc_key',  x['doc_key'])
            if doc_id == x['doc_key']:
                subtokens_mrg = list(itertools.chain.from_iterable(x['sentences']))

        span_texts = []
        for start_id, end_id in zip(start_ids, end_ids):
            # print(pprefix, 'max start', max(start_ids), 'max end', max(end_ids), 'len', len(subtokens_mrg))
            span_text = reformat_str(subtokens_mrg[int(start_id): int(end_id)+1])
            span_texts.append(span_text)
            # exit(0)

        num_clusters = max(cluster_ids)
        if num_clusters < 0:
            # only invalid spans
            continue

        cl_id_emb = defaultdict(list)
        for k, v in zip(cluster_ids, span_emb):
            cl_id_emb[int(k)].append(v)

        cl_id_spantext = defaultdict(list)
        for k, v in zip(cluster_ids, span_texts):
            cl_id_spantext[int(k)].append(v)

        for cl_id in cl_id_emb.keys():
            if cl_id < 0:
                continue
            spans = cl_id_emb[cl_id]
            cl_text = cl_id_spantext[cl_id]

            distances = cosine_distances(spans)
            # 1. pairwise average distance between valid spans that corefer
            cl_distance_sets['valid_coref'].append(np.average(distances))
            cl_distance_sets_text['valid_coref'].append(cl_text)

            # 2. pairwise average distance between valid spans that do not corefer
            valid_no_coref_spans = gather_values_excluding(cl_id_emb, cl_id)
            distances = cosine_distances(spans, valid_no_coref_spans)
            cl_distance_sets['valid_no_coref'].append(np.average(distances))
            cl_distance_sets_text['valid_no_coref'].append(cl_text)

            # 3. pairwise average distance between invalid span and valid span (per cluster)
            distances = cosine_distances(spans, cl_id_emb[-1])
            cl_distance_sets['invalid_no_coref'].append(np.average(distances))
            cl_distance_sets_text['invalid_no_coref'].append(cl_text)

        # 3. pairwise average distance between invalid span and valid span (per doc)
        valid_spans = gather_values_excluding(cl_id_emb, -1)
        distances = cosine_distances(valid_spans, cl_id_emb[-1])
        doc_distance_sets['invalid_valid'].append(np.average(distances))

        # 4. pairwise average distance between invalid spans
        distances = cosine_distances(cl_id_emb[-1])
        doc_distance_sets['invalid_invalid'].append(np.average(distances))

    results = {}

    for k, v in cl_distance_sets.items():
        print(pprefix, '[cluster level] average cosine distance', k, ':', np.average(np.array(v)))
        results['cl_'+k] = np.average(np.array(v))

    for k, v in doc_distance_sets.items():
        print(pprefix, '[doc level] average cosine distance', k, ':', np.average(np.array(v)))
        results['doc_'+k] = np.average(np.array(v))

    results_text = {}
    # for each cluster distance set, save the top n smallest and largest internal distance clusters
    for k in cl_distance_sets.keys():
        distance_text_pairs = list(map(lambda x, y: (x, y), cl_distance_sets[k], cl_distance_sets_text[k]))
        distance_text_pairs.sort(key = lambda x: x[0])
        if k == 'valid_coref':

            top_n = distance_text_pairs[:20]
            bottom_n = distance_text_pairs[-20:]
        else:
            top_n = distance_text_pairs[:n]
            bottom_n = distance_text_pairs[-n:]

        for i, x in enumerate(top_n):
            results_text['top'+str(i)+'_'+k +'_score'] = x[0]
            results_text['top'+str(i)+'_'+k +'_text'] = x[1]

        for i, x in enumerate(bottom_n):
            results_text['bot' + str(i) + '_' + k + '_score'] = x[0]
            results_text['bot' + str(i) + '_' + k + '_text'] = x[1]








    return results, results_text





    # loop through entities

    # TODO later add a few words from each entity so we know which ones are missed

def get_projection_id(projection_params):


    projection_name = 'ratio{}_maxex{}_rank{}_w{}_tp{}_dl{}'.format(projection_params['ratio'],
                                                              projection_params['maxex'],
                                                              projection_params['max_rank'],
                                                              projection_params['emb_weight'],
                                                              projection_params['to_project'],
                                                              projection_params['doc_level'])
    return projection_name

    # print('[get_conf_name] projetioN_name', projection_name)
def get_projection_params(yaml_args):
    emb_train_path = yaml_args['dataset']['embeddings']['train_path'][:-len('.h5')]
    maxex = -1
    ratio = -1.0
    param_list = emb_train_path.split('_')
    for param in param_list:
        if (param.startswith('maxex=')):
            maxex = param[len('maxex='):]
        if (param.startswith('ratio=')):
            ratio = param[len('ratio='):]
    max_rank = yaml_args['probe']['maximum_rank']
    emb_weight = yaml_args['dataset']['interpolation']
    if 'doc_level' in yaml_args['dataset']:
        projection = {'max_rank': max_rank,
                        'maxex': maxex,
                        'ratio': ratio,
                        'emb_weight': emb_weight,
                        'projection_path': proj_file,
                        'doc_level': yaml_args['dataset']['doc_level']
                          }
    else:
        projection = {'max_rank': max_rank,
                      'maxex': maxex,
                      'ratio': ratio,
                      'emb_weight': emb_weight,
                      'projection_path': proj_file,
                      'doc_level': ""
                      }
    return projection

if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('--project', default=1, type=int,
                      help='set 0 if dont want to project span embeddings')
    argp.add_argument('--proj_root', default="", type=str,
                      help='projection dir path, should contain directories of .yaml file and projection .npy file')
    argp.add_argument('--outfile', default="latent_eval_output.csv", type=str,
                      help='output file')
    argp.add_argument('--outdir', default="/projects/tir4/users/nmgandhi/coref/debug/latent_eval_output", type=str,
                      help='output directory')
    argp.add_argument('--span_emb_loc', default="", type=str,
                      help='held out set to evaluate on')
    argp.add_argument('--jsonlines_file', default="", type=str,
                      help='held out set jsonlines file')


    cli_args = argp.parse_args()
    np.random.seed(10)

    results_list = []
    results_text_list = []
    print('reading ', cli_args.jsonlines_file)
    input_jsonlines = load_jsonlines(cli_args.jsonlines_file)
    # yaml_file = glob.glob(cli_args.proj_dir + "*.yaml")[0]
    # projection_path = glob.glob(cli_args.proj_dir + "*.npy")[0]
    for proj_dir in os.listdir(cli_args.proj_root):
        if(len(glob.glob(os.path.join(cli_args.proj_root, proj_dir,'*.yaml'))) == 0 or
        len(glob.glob(os.path.join(cli_args.proj_root, proj_dir, '*.npy'))) == 0):
            continue
        yaml_file = glob.glob(os.path.join(cli_args.proj_root, proj_dir,'*.yaml'))[0]
        proj_file = glob.glob(os.path.join(cli_args.proj_root, proj_dir, '*.npy'))[0]
        yaml_args = yaml.load(open(yaml_file))
        projection_params = get_projection_params(yaml_args)
        projection_params['to_project'] = cli_args.project
        projection_id = get_projection_id(projection_params)

        # evaluating the test set
        start_indices_list, end_indices_list, cluster_ids_list, doc_embeddings_list, doc_ids_list = recover_span_repr(yaml_args,
                                                                                                        cli_args.span_emb_loc,
                                                                                                        projection_path=proj_file,
                                                                                                        to_project=cli_args.project)



        results, results_text = compute_distances(doc_embeddings_list, cluster_ids_list, start_indices_list, end_indices_list,
                                                  input_jsonlines, doc_ids_list)

        visualize_emb(cli_args, doc_embeddings_list, cluster_ids_list, start_indices_list, end_indices_list, input_jsonlines, doc_ids_list, projection_id)

        results['projection_id'] = projection_id
        results['dataset'] = cli_args.span_emb_loc
        results_text['projection_id'] = projection_id
        results_text['dataset'] = cli_args.span_emb_loc
        results_list.append(results)
        results_text_list.append(results_text)
        if not cli_args.project:
            break

    with open(cli_args.outfile, 'a') as f:
        writer = csv.writer(f)  # Note: writes lists, not dicts.
        for result in results_list:  # Maybe your df, or whatever iterable.
            writer.writerow(result)
    with open(cli_args.outfile, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, results_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(results_list)

    with open(cli_args.outfile[:-len('.csv')] + '_text.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, results_text_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(results_text_list)




