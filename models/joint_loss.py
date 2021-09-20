from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf

import util
import span_util
import coref_ops
import conll
import metrics
import optimization
from bert import tokenization
from bert import modeling
from pytorch_to_tf import load_from_pytorch_checkpoint
import sys
import itertools
from debug.span_analysis import *
import debug_util
from models.independent_concept_clutsers import CorefConceptModel
from emb_master import INCOMPAT_PAIRS, COMPAT_PAIRS
from collections import defaultdict
from sklearn.decomposition import PCA

class JointLossModel(CorefConceptModel):
    def __init__(self, config, args):
        self.args = args
        self.probe_rank = args['probe']['maximum_rank']
        self.word_pair_dims = (0, 1)
        self.meta_data = []

        super(JointLossModel, self).__init__(config)

    def get_candidate_labels_overlap(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):

        # overlap
        # label_start <= candidate_start <= label_end
        label_st_lt_cand_st = tf.less_equal(tf.expand_dims(labeled_starts, 1),
                                            tf.expand_dims(candidate_starts, 0))
        cand_st_lt_label_end = tf.greater_equal(tf.expand_dims(labeled_ends, 1),
                                                tf.expand_dims(candidate_starts, 0))
        start_overlap = tf.logical_and(label_st_lt_cand_st, cand_st_lt_label_end)  # [num_labeled, num_candidates]

        # label_start <= candidate_end <= label_end
        label_st_lt_cand_end = tf.less_equal(tf.expand_dims(labeled_starts, 1),
                                             tf.expand_dims(candidate_ends, 0))
        cand_end_lt_label_end = tf.greater_equal(tf.expand_dims(labeled_ends, 1),
                                                 tf.expand_dims(candidate_ends, 0))
        end_overlap = tf.logical_and(label_st_lt_cand_end, cand_end_lt_label_end)  # [num_labeled, num_candidates]

        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]

        num_labeled = util.shape(labels, 0)
        num_candidates = util.shape(candidate_starts, 0)

        # first, take the spans that have not had an exact match, and assign the concepts where the candidate start idx in the label
        same_span_counts = tf.math.count_nonzero(same_span, 0, dtype=tf.bool)
        no_span_match = tf.broadcast_to(tf.expand_dims(tf.logical_not(same_span_counts), 0),
                                        (num_labeled, num_candidates))
        start_overlap_mask = tf.logical_and(no_span_match, start_overlap)
        span_overlap = tf.logical_or(same_span, start_overlap_mask)

        # then, take the remaining spans that have not had an overlap match, and assign the concepts where the candidate end idx in the label
        overlap_span_counts = tf.math.count_nonzero(span_overlap, 0, dtype=tf.bool)
        no_span_overlap = tf.broadcast_to(tf.expand_dims(tf.logical_not(overlap_span_counts), 0),
                                          (num_labeled, num_candidates))
        end_overlap_mask = tf.logical_and(no_span_overlap, end_overlap)
        span_overlap2 = tf.logical_or(span_overlap, end_overlap_mask)

        # then apply the mask normal way

        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(span_overlap2))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]

        return candidate_labels

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends, k,
                     candidate_cluster_ids, candidate_concept_cluster_ids,
                     candidate_umls_cluster_ids):
        pprefix = '[get_span_emb]'
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs)  # [K, T]

            # projection loss
            if self.config['projection_loss_weight'] > 0:
                sample_num = 500
                ave_projection_loss = 0.0
                for i in range(100):
                    idxs = tf.range(tf.shape(head_attn_reps)[0])
                    ridxs = tf.random.shuffle(idxs)[:sample_num]
                    head_attn_reps_tr = tf.gather(head_attn_reps, ridxs)

                    candidate_cluster_ids_tr = tf.gather(candidate_cluster_ids, ridxs)
                    candidate_concept_cluster_ids_tr = tf.gather(candidate_concept_cluster_ids, ridxs)
                    candidate_umls_cluster_ids_tr = tf.gather(candidate_umls_cluster_ids, ridxs)

                    # PROJECTION LOSS
                    distance_pred = util.tf_cosine_distance(head_attn_reps_tr)
                    # distance_pred = self.squared_dist_predictions(transformed_head_attn_reps_tr)
                    # compute the labels
                    distance_labels = self.labels(head_attn_reps_tr, candidate_cluster_ids_tr,
                                                  candidate_concept_cluster_ids_tr, candidate_umls_cluster_ids_tr,
                                                  k)
                    # compute the projection loss
                    projection_loss = self.l1_projection_loss(distance_labels, distance_pred, k)
                    ave_projection_loss += projection_loss

                ave_projection_loss /= 10.0
            else:
                ave_projection_loss = tf.constant(0.0)

            # scaffolding loss (ONLY i2b2)
            if self.config['scaffolding_loss_weight'] > 0:
                if self.config['binary_i2b2']:
                    num_labels = 2
                else:
                    num_labels = 6

                concept_scaffolding_weights = tf.get_variable("concept_weights_scaffolding", [util.shape(context_outputs,1), num_labels], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=True) # [T, |c|]
                concept_weights = tf.matmul(head_attn_reps, concept_scaffolding_weights) # [K, |c|]
                sparse_labels = tf.reshape(candidate_concept_cluster_ids, [-1, 1])
                derived_size = tf.shape(candidate_concept_cluster_ids)[0]
                indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
                concated = tf.concat([indices, sparse_labels], 1)
                outshape = tf.stack([derived_size, num_labels])
                concept_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

                scaffolding_loss = tf.reduce_sum(self.softmax_loss(concept_weights, concept_labels))
            else:
                scaffolding_loss = tf.constant(0.0)
            # check if concat or replace
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]


        return span_emb, ave_projection_loss, scaffolding_loss, head_attn_reps  # [k, emb]



    def labels(self, span_emb, cluster_ids, concept_ids, umls_ids, num_candidates):
        """Computes the distances between all pairs of spans; returns them as a torch tensor.

        Args:
          observation: a single Observation class for a document:
        Returns:
          A torch tensor of shape (num_spans, num_spans) of distances
          in the coref clusters as specified by the observation annotation.
        """

        pprefix = '[labels]'
        coref_distances = util.custom_distance_coref(cluster_ids)
        concept_distances = util.custom_distance_concepts(concept_ids)
        umls_distances = util.custom_distance_coref(umls_ids)

        concept_weight = self.args['dataset']['concept_weight']
        coref_weight = self.args['dataset']['coref_weight']
        umls_weight = self.args['dataset']['umls_weight']

        distances = (coref_weight * coref_distances) + (concept_weight * concept_distances) + (umls_weight * umls_distances)

        return distances

    def squared_dist_predictions(self, transformed):

        r = tf.reduce_sum(transformed * transformed, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2 * tf.matmul(transformed, tf.transpose(transformed)) + tf.transpose(r)
        return D



    def l1_projection_loss(self, label_batch, predictions, length_batch):
        # length_batch = predictions.get_shape().as_list()[0]
        labels_1s = tf.to_float(tf.not_equal(label_batch, tf.constant(-1.0)))
        predictions_masked = tf.to_float(predictions) * tf.to_float(labels_1s)
        labels_masked = label_batch * labels_1s
        total_sents = tf.constant(1.0)
            # tf.reduce_sum(tf.to_float(tf.not_equal(tf.constant(length_batch), tf.constant(0))))

        sq = tf.pow(length_batch,2)
        squared_lengths = tf.to_float(sq)
        # if total_sents > 0:
        loss_per_sent = tf.reduce_sum(tf.abs(predictions_masked - labels_masked), axis=self.word_pair_dims)
        normalized_loss_per_sent = loss_per_sent / squared_lengths
        batch_loss = tf.reduce_sum(normalized_loss_per_sent) / total_sents
        # else:
        #     batch_loss = tf.constant(0.0)
        return batch_loss#,total_sents

    def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts,
                                 gold_ends, cluster_ids, concept_starts, concept_ends, concept_cluster_ids,
                                 umls_starts, umls_ends, umls_cluster_ids,
                                 sentence_map):
    #
    #
    # def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, concept_ids, genre, is_training, gold_starts,
    #                              gold_ends, cluster_ids, sentence_map):
        pprefix = '[get_predictions_and_loss]'

        print(pprefix, 'type(sentence_map)', type(sentence_map), sentence_map.shape)
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        all_encoder_layers = model.get_all_encoder_layers()
        mention_doc = model.get_sequence_output()  # [batch_size, seq_length, hidden_size]
        print(pprefix, 'mention_doc', mention_doc.shape.as_list())

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        num_sentences = tf.shape(mention_doc)[0]
        max_sentence_length = tf.shape(mention_doc)[1]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
        num_words = util.shape(mention_doc, 0)
        k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))

        antecedent_doc = mention_doc

        flattened_sentence_indices = sentence_map
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
                                                     flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]

        candidate_concept_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, concept_starts,
                                                              concept_ends,
                                                              concept_cluster_ids)  # [num_candidates]


        # convert concept_ids to mention detection task
        if self.config['binary_i2b2']:
            candidate_concept_cluster_ids = tf.to_int32(tf.not_equal(candidate_concept_cluster_ids, tf.zeros_like(candidate_concept_cluster_ids)))

        if self.config['overlap_umls']:
            candidate_umls_cluster_ids = self.get_candidate_labels_overlap(candidate_starts, candidate_ends, umls_starts,
                                                                           umls_ends,
                                                                           umls_cluster_ids)  # [num_candidates]

        else:
            candidate_umls_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, umls_starts,
                                                                  umls_ends,
                                                                  umls_cluster_ids)  # [num_candidates]

        candidate_span_emb, projection_loss, scaf_loss, candidate_head_attn_reps = self.get_span_emb(mention_doc, mention_doc, candidate_starts,
                                                                candidate_ends, k, candidate_cluster_ids,
                                                                candidate_concept_cluster_ids, candidate_umls_cluster_ids)  # [num_candidates, emb]



        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        # beam size
        c = tf.minimum(self.config["max_top_antecedents"], k)
        print(pprefix, 'k', k)
        # pull from beam
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   num_words,
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]

        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        # top_span_concept_ids = tf.gather(concept_ids, top_span_starts)  # [k]i




        # TODO: gather the top concept ids
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02)), genre)  # [emb]
        if self.config['use_metadata']:
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
            top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]i
        else:
            top_span_speaker_ids = None




        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
            top_span_emb, top_span_mention_scores, c)
        num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
        word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
        flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
        mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1)  # [k, 1]
        antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents))  # [k, c]
        segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0,
                                            self.config['max_training_sentences'] - 1) if self.config[
            'use_segment_distance'] else None  # [k, c]

        if self.config['fine_grained']:
            for i in range(self.config["coref_depth"]):
                with tf.variable_scope("coref_layer", reuse=(i > 0)):
                    top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                    top_antecedent_scores_ = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                          top_antecedents,
                                                                                                          top_antecedent_emb,
                                                                                                          top_antecedent_offsets,
                                                                                                          top_span_speaker_ids,
                                                                                                          genre_emb,
                                                                                                          segment_distance)  # [k, c]
                    top_antecedent_weights = tf.nn.softmax(
                        tf.concat([dummy_scores, top_antecedent_scores_], 1))  # [k, c + 1]
                    top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                                   1)  # [k, c + 1, emb]
                    attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                      1)  # [k, emb]
                    with tf.variable_scope("f"):
                        f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                       util.shape(top_span_emb, -1)))  # [k, emb]
                        top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]
        else:
            top_antecedent_scores_ = top_fast_antecedent_scores  # [k, c]


        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores_], 1)  # [k, c + 1]




        ## COREF LOSS
        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]
        coref_loss = tf.reduce_sum(loss)  # []

        joint_loss = self.config['coref_loss_weight']*coref_loss+ self.config["projection_loss_weight"]*projection_loss + self.config['scaffolding_loss_weight']*scaf_loss

        return [candidate_starts, candidate_ends, candidate_mention_scores, candidate_head_attn_reps, top_span_starts,
            top_span_ends, top_antecedents, top_antecedent_scores, projection_loss], (joint_loss, scaf_loss), 0


    def evaluate(self, session, model_id, global_step=None, official_stdout=False, keys=None, eval_mode=False):
        pprefix = '[evaluate]'
        self.load_eval_data(eval_mode=eval_mode)

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()
        losses = []
        doc_keys = []
        num_evaluated = 0
        span_info_list = []
        gold_span_info_list = []
        cl_id_count = 0

        jsonlines_predicted = []
        ave_projection_loss = 0.0

        incompat_texts = list(itertools.chain(*INCOMPAT_PAIRS))
        compat_texts = list(itertools.chain(*COMPAT_PAIRS))
        incompat_distances = defaultdict(list)
        compat_distances = defaultdict(list)

        span_xy_pairs = debug_util.load_pickle(self.config['projection_pairs_path'])
        span_projections = {}


        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            # _, _, _, _, _, _, gold_starts, gold_ends,cluster_ids, _ = tensorized_example
            _, _, _, _, _, _, gold_starts, gold_ends,cluster_ids, concept_starts, concept_ends,_,_,_,_, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            # if tensorized_example[0].shape[0] <= 9:
            if keys is not None and example['doc_key'] not in keys:
                print('Skipping...', example['doc_key'], tensorized_example[0].shape)
                continue
            doc_keys.append(example['doc_key'])
            loss, (candidate_starts, candidate_ends, candidate_mention_scores, candidate_span_emb, top_span_starts,
                top_span_ends, top_antecedents,
                top_antecedent_scores, projection_loss) = session.run([self.loss, self.predictions], feed_dict=feed_dict)

            subtokens_mrg = list(itertools.chain.from_iterable(example['sentences']))
            incompat_distances = debug_util.update_span_distances(candidate_starts, candidate_ends,
                                                                  candidate_span_emb, subtokens_mrg,
                                                                  incompat_texts, INCOMPAT_PAIRS,
                                                                  incompat_distances)
            compat_distances = debug_util.update_span_distances(candidate_starts, candidate_ends,
                                                                candidate_span_emb, subtokens_mrg,
                                                                compat_texts, COMPAT_PAIRS,
                                                                compat_distances)

            doc_key = example['doc_key']
            span_projections[doc_key] = debug_util.update_span_projections(candidate_starts,
                                                                           candidate_ends,
                                                                           candidate_span_emb,
                                                                           span_xy_pairs[doc_key],
                                                                           doc_key)

            print('span_projections[doc_key]', span_projections[doc_key])
            ave_projection_loss += projection_loss
            for name, x in zip(['candidate_starts', 'candidate_ends', 'candidate_mention_scores',
                                'top_span_starts', 'top_span_ends', 'top_antecedents', 'top_antecedent_scores'],
                               [candidate_starts, candidate_ends, candidate_mention_scores,
                                top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores]):
                if type(x) == np.ndarray:
                    x = x.tolist()
                # if hasattr(x, 'shape'):
                #     print(pprefix, name, type(name), x.shape)
                # else:
                #     print(pprefix, name, type(name))
                example[name] = x

            cl_id_count += len(example['clusters'])

            losses.append(loss)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            # print(pprefix, 'predicted_antecedents', len(predicted_antecedents), predicted_antecedents)
            # print(pprefix, 'example["clusters"]', example["clusters"])

            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)

            example['predicted_clusters'] = coref_predictions[example["doc_key"]]
            jsonlines_predicted.append(example)

            if eval_mode:
                output_id = 'partition=test' + '_dataset=' + (
                    'tgt' if 'i2b2' in self.config['eval_path'] else 'src') + '_maxex=' + str(len(self.eval_data))
                output_file = 'model=' + model_id + '_' + output_id + '.h5'
                out_filename = os.path.join(self.config['emb_dir'], output_file)
                print(pprefix, 'candidate_span_emb', candidate_span_emb.shape)

                # span_util.save_emb(example, candidate_span_emb, candidate_starts, candidate_ends, [],
                #                    cl_id_count, out_filename, example_num)
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        for k in incompat_distances:
            print('[incompat_distances] {}:{}'.format(k, np.array(incompat_distances[k]).mean()))
        # print('compat_distances', compat_distances)
        for k in compat_distances:
            print('[compat_distances] {}:{}'.format(k, np.array(compat_distances[k]).mean()))

        print('span pair projections: (doc key {})'.format(doc_keys[0]), span_projections[doc_keys[0]])
        projections_flattened = []
        span_pair_names = []
        x_emb_flattened = []
        for k in span_projections.keys():
            span_pair_names += [span_pair['proj_name'] for span_pair in span_projections[k]]
            projections_flattened += [span_pair['projection'].flatten() for span_pair in span_projections[k]]
            x_emb_flattened += [span_pair['x_emb'].flatten() for span_pair in span_projections[k]]

        projections_flattened = np.array(projections_flattened)
        # projections_reduced = pca_model.fit_transform(projections_flattened)
        debug_util.save_pickle({'projections_flattened':projections_flattened,
                                'proj_name':span_pair_names,
                                'x_emb_flattened':x_emb_flattened}, 'joint_model_span_pairs_proj_flattened.pickle')
        # for i, (span_pair_name, proj) in enumerate(zip(span_pair_names, projections_reduced)):
        #     print('{}: span pair name:{}, proj:{}'.format(i, span_pair_name, proj))





        summary_dict = {}
        # if eval_mode:
        average_f1 = 0.0
        if True:
            print('[evaluate] self.config["conll_eval_path"]', self.config["conll_eval_path"])
            conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, self.subtoken_maps,
                                                 output_conll_file=os.path.join(self.config["eval_path"],
                                                                                self.config['evaluation_dir'],
                                                                                '{}.conll'.format(model_id)),
                                                 official_stdout=official_stdout)
            # nested dict for each eval metric

            average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            summary_dict["Average F1 (conll)"] = average_f1
            print("Average F1 (conll): {:.2f}%".format(average_f1))

            # add conditional to not save the jsonlines file

            # util.save_results(conll_results, model_id, self.config["eval_path"], self.config['evaluation_dir'],
            #                   self.config['evaluation_file'], jsonlines_predicted)

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = average_f1

        # summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        ave_projection_loss /= float(len(self.eval_data))
        summary_dict['projection_loss'] = [ave_projection_loss]

        return summary_dict, f
