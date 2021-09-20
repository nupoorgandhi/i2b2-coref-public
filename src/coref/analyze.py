from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

import tensorflow as tf
import numpy as np
import span_util
import util
import h5py
from models.custom_coref import CustomCorefIndependent
from tqdm import tqdm
import re
import random
from debug.span_analysis import SpanAnalysis








if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    # Input file in .jsonlines format.
    src_input_filename = '/usr1/home/nmgandhi/dhs/coref-orig/output_dir/train.i2b2.english.384.jsonlines'
    tgt_input_filename = '/usr1/home/nmgandhi/dhs/coref-orig/output_dir/train.english.384.jsonlines'


    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    proj_src_dir =  '/usr1/home/nmgandhi/dhs/coref-orig/debug/src_proj_interp=.5/train=onto_test=onto/'

    proj_src_tgt_dir =  '/usr1/home/nmgandhi/dhs/coref-orig/debug/src_tgt_proj_interp=.5/train=onto_test=i2b2/'
    for outdir in [proj_src_dir, proj_src_tgt_dir]:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for sub_dir in ['span_emb_extrema_distances','span_emb_clusters']:
            joint = os.path.join(outdir, sub_dir)
            if not os.path.exists(joint):
                os.makedirs(joint)


    proj_src =  '/usr1/home/nmgandhi/dhs/structural-probes/reporting/interp=.5/spanBERT-disk-coref-distance-2021-4-18-14-52-32-105008/predictor.params.npy'
    proj_src_tgt = '/usr1/home/nmgandhi/dhs/structural-probes/reporting_i2b2_transformed/interp=.5/spanBERT-disk-coref-distance-2021-4-18-18-13-49-978562/predictor.params.npy'



    analyzer_src = SpanAnalysis(proj_src_dir, ['i2b2','onto'])
    analyzer_src_tgt = SpanAnalysis(proj_src_tgt_dir, ['i2b2','onto'])

    # exp1: ontonotes, look at span projection with the sourse transofrmation applied
    # exp2: ontonotes, look at span projection with the source and target transformation applied
    # exp3: i2b2, look at span projection with the sourse transofrmation applied
    # exp4: i2b2, look at the span projection with the source and target transformation applied

    with tf.Session() as session:
        model.restore(session)

        with open(src_input_filename) as input_file:



            print('[input_file]', src_input_filename)
            parent_child_list = []
            candidate_span_list = []

            doc_keys = []

            write_count = 0

            num_lines = sum(1 for line in input_file.readlines())
            input_file.seek(0)  # return to first line
            # print('at beginning of file', len(input_file.readlines()))
            lines = input_file.readlines()
            random.shuffle(lines)
            for example_num, line in enumerate(tqdm(lines[:100])):
                # print('in loop:')
                example = json.loads(line)
                doc_keys.append(example['doc_key'])

                tensorized_example = model.tensorize_example(example, is_training=False)
                feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)

                # print('candidate_span_emb:', candidate_span_emb.shape)
                # print('candidate_starts:', candidate_starts.shape)

                # apply the source projection
                candidate_span_emb_src = span_util.get_emb(example['clusters'], candidate_span_emb, candidate_starts,
                                                        candidate_ends, balanced=False,
                                                        project_paths=[proj_src], info=False).eval()
                analyzer_src.load_embeddings(example['clusters'], candidate_span_emb_src, example['sentences'],[],
                                             candidate_starts, candidate_ends, example['doc_key'], domain='onto')



                # apply source -> source,target projection
                candidate_span_emb_src_tgt = span_util.get_emb(example['clusters'], candidate_span_emb, candidate_starts,
                                                        candidate_ends, balanced=False,
                                                        project_paths=[proj_src, proj_src_tgt], info=False).eval()
                analyzer_src_tgt.load_embeddings(example['clusters'], candidate_span_emb_src_tgt, example['sentences'], [],
                                             candidate_starts, candidate_ends, example['doc_key'], domain='onto')

        with open(tgt_input_filename) as input_file:


            print('[input_file]', tgt_input_filename)
            parent_child_list = []
            candidate_span_list = []

            doc_keys = []

            write_count = 0

            num_lines = sum(1 for line in input_file.readlines())
            input_file.seek(0)  # return to first line
            # print('at beginning of file', len(input_file.readlines()))
            lines = input_file.readlines()
            random.shuffle(lines)
            for example_num, line in enumerate(tqdm(lines[:100])):
                # print('in loop:')
                example = json.loads(line)
                doc_keys.append(example['doc_key'])

                tensorized_example = model.tensorize_example(example, is_training=False)
                feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
                candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings,
                                                                                   feed_dict=feed_dict)

                # print('candidate_span_emb:', candidate_span_emb.shape)
                # print('candidate_starts:', candidate_starts.shape)

                # apply the source projection
                candidate_span_emb_src = span_util.get_emb(example['clusters'], candidate_span_emb, candidate_starts,
                                                           candidate_ends, balanced=False,
                                                           project_paths=[proj_src], info=False).eval()
                analyzer_src.load_embeddings(example['clusters'], candidate_span_emb_src, example['sentences'], [],
                                             candidate_starts, candidate_ends, example['doc_key'], domain='i2b2')


                # apply source -> source,target projection
                candidate_span_emb_src_tgt = span_util.get_emb(example['clusters'], candidate_span_emb,
                                                               candidate_starts,
                                                               candidate_ends, balanced=False,
                                                               project_paths=[proj_src, proj_src_tgt], info=False).eval()
                analyzer_src_tgt.load_embeddings(example['clusters'], candidate_span_emb_src_tgt, example['sentences'],
                                                 [],
                                                 candidate_starts, candidate_ends, example['doc_key'], domain='i2b2')

        analyzer_src.write_clusters()
        analyzer_src.embedding_distances()
        analyzer_src_tgt.write_clusters()
        analyzer_src_tgt.embedding_distances()








