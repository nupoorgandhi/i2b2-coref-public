from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import itertools
import os, sys
import h5py



def save_emb(example, candidate_span_emb, candidate_starts, candidate_ends, projection_paths, cl_id_count,
             out_filename, example_num):
    candidate_span_info = get_emb(example['predicted_clusters'] + example['clusters'],
                                            candidate_span_emb,
                                            candidate_starts,
                                            candidate_ends,
                                            example['token_ids'],
                                            project_paths=projection_paths,
                                            balanced=True, info=True,
                                            cl_id_count=cl_id_count)


    sys.stdout.flush()
    with h5py.File(out_filename, 'a') as hf:
        # print('writing to', out_filename)

        span_emb = candidate_span_info.eval()

        print('span_emb shape', span_emb.shape)
        if 'src' in out_filename:
            hf.create_dataset(str(example_num), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])),
                              compression="gzip", compression_opts=0, shuffle=False, chunks=True)
            # print('span_emb', span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])).shape, type(span_emb))
        else:
            # print('doc_key', example['doc_key'])
            hf.create_dataset(str(example['doc_key']), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])),
                              compression="gzip", compression_opts=0, shuffle=False, chunks=True)

def cluster_idx(start_idx, end_idx, clusters):

    for cl_idx, cl in enumerate(clusters):
        if (start_idx, end_idx) in cl:
            return cl_idx
    return -1

def get_concept(start_idx, end_idx, concepts):
    span_concepts = []
    for i in range(start_idx, end_idx+1):
        if int(concepts[i]) > 0:
            return concepts[i]
    return 0


def recover_projection(proj_path):
    pprefix = '[recover_projection]'
    print(pprefix, 'proj_path', proj_path)
    return np.load(proj_path)


def project_span_torch(batch, proj):
    pprefix = '[project_span]'
    if(type(batch) == np.ndarray):
        batch = torch.from_numpy(batch) # (max_seq_len, representation_dim)
    if(type(proj) == np.ndarray):
        proj = torch.from_numpy(proj) # (representation_dim, rank)
    # print(pprefix, 'batch', batch.shape, type(batch), 'proj', proj.shape, type(proj))
    transformed = torch.matmul(batch, proj)
    return transformed.detach().numpy()

def project_span_tf(batch, proj):
    pprefix = '[project_span]'
    # batch = torch.from_numpy(batch)  # (max_seq_len, representation_dim)
    # proj = torch.from_numpy(proj)  # (representation_dim, rank)

    # print(pprefix, 'batch', batch.shape, type(batch), 'proj', proj.shape, type(proj))
    transformed = tf.linalg.matmul(batch, proj)
    return transformed
    # batchlen, seqlen, rank = transformed.size()
    # print(pprefix, 'seqlen, rank', seqlen, rank)
    # transformed = transformed.unsqueeze(2)
    # transformed = transformed.expand(-1, -1, seqlen, -1)
    # print(pprefix, 'transformed', transformed.shape)
    # return transformed.detach().numpy()


def get_emb(clusters, cand_span_emb, span_starts, span_ends, concepts, balanced=True, project_paths=[], info=True, cl_id_count=0):
    """
    Make [N, emb + 3] tensor where N is the number of candidate spans, and emb is dimension of
    embeddings and last 3 values are start_idx, end_idx, cluster_id (local to document)
    :param clusters: dict
    :param span_emb: [N,2324]
    :param span_starts: [N,]
    :param span_ends: [N,]
    :return:
    """
    pprefix = '[get_emb]'
    span_emb_list = []
    cl_idx_list = []
    span_concepts = []
    num_pos = 0
    num_neg = 0
    # TODO randomize order
    mask = np.ones((len(span_starts)))
    concepts = list(itertools.chain.from_iterable(concepts))
    for i, (start_idx, end_idx) in enumerate(zip(span_starts, span_ends)):
        # emb = cand_span_emb[i]
        # determine if span belongs to a gold cluster, -1 if not
        cl_idx = cluster_idx(start_idx, end_idx, clusters)

        if cl_idx == -1:
            num_neg += 1
        else:
            num_pos += 1
            cl_idx += cl_id_count

        # span_emb_list.append(emb)
        # print(pprefix, 'emb shape', emb.shape)
        # exit(0)
        cl_idx_list.append(cl_idx)
        span_concepts.append(get_concept(start_idx, end_idx, concepts))

        if (num_neg > 10+ num_pos and cl_idx == -1):
            mask[i] = False

    # print(pprefix, 'cluster idx frq', pd.Series(cl_idx_list).value_counts())
    # print(pprefix, 'span_meb_list len', len(span_emb_list))

    span_starts = tf.dtypes.cast(tf.stack(span_starts, 0), tf.float32)
    span_starts = tf.reshape(span_starts, [-1, 1])

    span_ends = tf.dtypes.cast(tf.stack(span_ends, 0), tf.float32)
    span_ends = tf.reshape(span_ends, [-1, 1])

    span_concepts = tf.dtypes.cast(tf.stack(span_concepts, 0), tf.float32)
    span_concepts = tf.reshape(span_concepts, [-1, 1])

    cl_idx = tf.dtypes.cast(tf.stack(cl_idx_list, 0), tf.float32)
    cl_idx = tf.reshape(cl_idx, [-1, 1])

    for proj_path in project_paths:
        # print(pprefix, 'projecting', proj_path)
        proj = recover_projection(proj_path=proj_path)
        cand_span_emb = project_span_torch(cand_span_emb, proj)

    if balanced:
        # print(pprefix, 'before masking shape:', tf.concat([cand_span_emb, span_starts, span_ends, cl_idx], 1).shape)
        # print(pprefix, 'after masking shape:',
        #       tf.boolean_mask(tf.concat([cand_span_emb, span_starts, span_ends, cl_idx], 1), mask).shape)
        if info:
            # print(pprefix, 'shape', tf.boolean_mask(tf.concat([cand_span_emb, span_starts, span_ends, span_concepts, cl_idx], 1), mask).shape)

            return tf.boolean_mask(tf.concat([cand_span_emb, span_starts, span_ends, span_concepts, cl_idx], 1), mask)
        else:
            return tf.boolean_mask(cand_span_emb, mask)
    else:
        if info:
            return tf.concat([cand_span_emb, span_starts, span_ends, span_concepts, cl_idx], 1)
        else:
            return cand_span_emb

    pass
