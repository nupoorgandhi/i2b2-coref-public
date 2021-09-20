import sys
import itertools
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import collections
import os
#TODO

class SpanAnalysis(object):
    def __init__(self, directory, domains=['i2b2', 'onto']):


        self.directory = directory

        # store embedding span representations
        self.TN = {}
        self.TP = {}

        self.TP_span_text = {}
        self.TP_doc_id = {}
        self.TP_starts= {}
        self.TP_sentences= {}
        self.TP_ends= {}

        self.TN_span_text= {}
        self.TN_doc_id= {}
        self.TN_starts= {}
        self.TN_ends= {}
        self.TN_sentences= {}


        for d in domains:
            self.TN[d] = np.array([])
            self.TP[d] = np.array([])

            self.TP_span_text[d] = []
            self.TP_doc_id[d] = []
            self.TP_starts[d] = []
            self.TP_sentences[d] = []
            self.TP_ends[d] = []

            self.TN_span_text[d] = []
            self.TN_doc_id[d] = []
            self.TN_starts[d] = []
            self.TN_ends[d] = []
            self.TN_sentences[d] = []








    def ave_embedding_similarity(self, arr1, arr2=np.array([])):
        pprefix = '[ave_embedding_similarity]'
        # print(pprefix, 'arrs[0]', arr1.shape)
        if(arr1.size == 0):
            return -1
        if(arr2.size == 0):
            return np.average(cosine_similarity(arr1))
        else:
            return np.average(cosine_similarity(arr1, arr2))
        print(pprefix, 'np.average(arrs[0], axis=0)', np.average(arrs[0], axis=0).shape)
        l = arrs[0].shape[1]

        ave_embs = np.concatenate((np.average(arrs[0], axis=0).reshape((1, l)), np.average(arrs[1], axis=0).reshape((1, l))), axis=0)
        print('[ave_embedding_similarity]shape:', ave_embs.shape)
        # return np.dot(ave_embs[0], ave_embs[1])

        return cosine_similarity(ave_embs[0].reshape((1,l)), ave_embs[1].reshape((1,l)))

    def reformat_str(self, subtoken_list):
        subtoken_str = ''
        for i, x in enumerate(subtoken_list):
            if (x[0] == '#'):
                subtoken_str += x
            else:
                subtoken_str += (" " + x)
        return subtoken_str.replace('#', '')
    def k_largest_index(self, a, k):
        idx = np.argpartition(-a.ravel(), k)[:k]
        return np.column_stack(np.unravel_index(idx, a.shape))

    def most_representative_examples(self, span_emb, starts, ends, sentences, d, filename):
        pprefix = '[most_representative_examples]'
        print('\n', pprefix, 'd:',d)
        if(span_emb.size == 0):
            return
        # cluster the spans, and find the most representative examples
        k = 100
        km = KMeans(n_clusters=k)
        X_distances = km.fit_transform(span_emb)
        label_counts = collections.Counter(km.predict(span_emb))

        with open(filename, 'w') as writer:
            for i in range(k):
                # print(pprefix, '-X_distances[:,i]', X_distances[:,i].shape)
                indices = np.argpartition(X_distances[:,i].ravel(), min(X_distances.shape[0],10))[:min(X_distances.shape[0],10)]
                # print(pprefix, 'indices', indices, 'X_distances', X_distances.shape)
                writer.write(' '.join([pprefix, 'cluster', str(i), 'size', str(label_counts[i]), ' '.join([self.reformat_str(sentences[j][starts[j]:ends[j]+1]) for j in indices])]) + "\n")





    def extrema_distance_examples(self, span_emb1, span_emb2, starts1, starts2, ends1, ends2, sentences1,
                                  sentences2, span_text1, span_text2, doc_id1, doc_id2, d1, d2):


        pprefix = '[extrema_distance_examples]'
        directory = os.path.join(self.directory, 'span_emb_extrema_distances')
        filename = os.path.join(directory, '_'.join([d1,d2]) + '.txt')


        print('\n', pprefix, 'd1', d1, ', d2', d2)
        k = 30
        c = 20
        lines = []


        sim_matrix = cosine_similarity(span_emb1, span_emb2)
        idx = np.argpartition(-sim_matrix.ravel(), k)[:k]
        max_indices = np.column_stack(np.unravel_index(idx, sim_matrix.shape))
        # lines.append(' '.join([pprefix, 'max_indices', ' '.join(max_indices)]))
        for i, x in enumerate(max_indices):
            lines.append(' '.join([pprefix, 'example ', str(i)]))
            idx1 = x[0]
            idx2 = x[1]
            # print(pprefix, 'idx1', idx1, 'idx2', idx2)
            # print(pprefix, 'sentence indices 1', max(0, starts1[idx1]-5), min(len(sentences1[idx1]), ends1[idx1]+5))
            # print(pprefix, 'sentence indices 2', max(0, starts2[idx2]-5),min(len(sentences2[idx2]), ends2[idx2]+5))
            lines.append(' '.join([pprefix, d1, 'span:', span_text1[idx1],
                  'context:', self.reformat_str(sentences1[idx1][max(0, starts1[idx1]-c): min(len(sentences1[idx1]), ends1[idx1]+c)])]))
            lines.append(' '.join([pprefix, d2, 'span:', span_text2[idx2],
                  'context:', self.reformat_str(sentences2[idx2][max(0, starts2[idx2]-c):min(len(sentences2[idx2]), ends2[idx2]+c)])]))
            lines.append('\n')

        idx = np.argpartition(sim_matrix.ravel(), k)[:k]
        min_indices = np.column_stack(np.unravel_index(idx, sim_matrix.shape))
        # print(pprefix, 'min_indices', min_indices)
        for i, x in enumerate(min_indices):
            lines.append(' '.join([pprefix, 'example ', str(i)]))
            idx1 = x[0]
            idx2 = x[1]
            # print(pprefix, 'idx1', idx1, 'idx2', idx2)
            # print(pprefix, 'sentence indices', max(0, starts1[idx1]-5), min(len(sentences1), ends1[idx1]+5))

            lines.append(' '.join([pprefix, d1, 'span:', span_text1[idx1],
                  'context:', self.reformat_str(sentences1[idx1][max(0, starts1[idx1] - c): min(len(sentences1[idx1]), ends1[idx1] + c)])]))
            lines.append(' '.join([pprefix, d2, 'span:', span_text2[idx2],
                  'context:', self.reformat_str(sentences2[idx2][max(0, starts2[idx2] - c): min(len(sentences2[idx2]), ends2[idx2] + c)])]))
            lines.append('\n')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))


    def write_clusters(self):
        pprefix = '[write_clusters]'
        directory = os.path.join(self.directory, 'span_emb_clusters')
        for segment in ['TN', 'TP']:
            for d in ['i2b2', 'onto']:
                span_emb = self.TN[d] if (segment == 'TN') else self.TP[d]
                starts = self.TN_starts[d] if (segment == 'TN') else self.TP_starts[d]
                ends = self.TN_ends[d] if (segment == 'TN') else self.TP_ends[d]
                sentences = self.TN_sentences[d] if (segment == 'TN') else self.TP_sentences[d]
                filename = os.path.join(directory, '_'.join([segment, d]) + '.txt')
                self.most_representative_examples(span_emb, starts, ends, sentences, d, filename)
            span_emb = np.concatenate((self.TN['i2b2'],self.TN['onto']), axis=0) if (segment == 'TN') else np.concatenate((self.TP['i2b2'],self.TP['onto']), axis=0)
            starts = (self.TN_starts['i2b2'] + self.TN_starts['onto']) if (segment == 'TN') else (self.TP_starts['i2b2'] + self.TP_starts['onto'])
            ends = (self.TN_ends['i2b2'] + self.TN_ends['onto']) if (segment == 'TN') else (self.TP_ends['i2b2'] + self.TP_ends['onto'])
            sentences = (self.TN_sentences['i2b2'] + self.TN_sentences['onto']) if (segment == 'TN') else (self.TP_sentences['i2b2'] + self.TP_sentences['onto'])
            filename = os.path.join(directory, segment + '.txt')
            self.most_representative_examples(span_emb, starts, ends, sentences, d, filename)

        for d in ['i2b2', 'onto']:


            span_emb_shape = self.TN[d].shape if len( self.TN[d].shape) > len(self.TP[d].shape) else self.TP[d].shape
            shape_2 = 0 if len(self.TN[d].shape) < 2 else self.TN[d].shape[1]
            self.TN[d] = self.TN[d].reshape((self.TN[d].shape[0],span_emb_shape[1]))
            shape_2 = 0 if len(self.TP[d].shape) < 2 else self.TN[d].shape[1]
            self.TP[d] = self.TP[d].reshape((self.TP[d].shape[0],span_emb_shape[1]))
            print(pprefix, 'self.TP[d].shape', self.TP[d].shape)
            print(pprefix, 'self.TN[d].shape', self.TN[d].shape)


            span_emb = np.concatenate((self.TN[d], self.TP[d]), axis=0)
            starts = self.TN_starts[d] + self.TP_starts[d]
            ends = self.TN_ends[d] + self.TP_ends[d]
            sentences = self.TN_sentences[d] + self.TP_sentences[d]
            filename = os.path.join(directory, d + '.txt')
            self.most_representative_examples(span_emb, starts, ends, sentences, d, filename)





    def embedding_distances(self):
        pprefix = '[embedding_distances]'


        print(pprefix, 'TN:\n')
        print(pprefix, 'source <-> target', self.ave_embedding_similarity(self.TN['onto'], self.TN['i2b2']))


        self.extrema_distance_examples(self.TN['onto'], self.TN['i2b2'], self.TN_starts['onto'], self.TN_starts['i2b2'],
                                       self.TN_ends['onto'], self.TN_ends['i2b2'], self.TN_sentences['onto'],
                                       self.TN_sentences['i2b2'], self.TN_span_text['onto'], self.TN_span_text['i2b2'] ,
                                       self.TN_doc_id['onto'], self.TN_doc_id['i2b2'], 'TN_onto', 'TN_i2b2')
        # print(pprefix, 'TP:\n')
        # print(pprefix, 'source <-> target',self.ave_embedding_similarity(self.TP['onto'], self.TP['i2b2']))
        #
        # self.extrema_distance_examples(self.TP['onto'], self.TP['i2b2'], self.TP_starts['onto'], self.TP_starts['i2b2'],
        #                                self.TP_ends['onto'], self.TP_ends['i2b2'], self.TP_sentences['onto'],
        #                                self.TP_sentences['i2b2'], self.TP_span_text['onto'], self.TP_span_text['i2b2'],
        #                                self.TP_doc_id['onto'], self.TP_doc_id['i2b2'], 'TP_onto', 'TP_i2b2')
        #

        #
        #
        # distance from source embeddings
        source_all = np.concatenate((self.TN['onto'], self.TP['onto']), axis=0)
        source_all_starts = self.TN_starts['onto'] + self.TP_starts['onto']
        source_all_ends = self.TN_ends['onto'] + self.TP_ends['onto']
        source_all_sentences = self.TN_sentences['onto'] + self.TP_sentences['onto']
        source_all_spans = self.TN_span_text['onto'] + self.TP_span_text['onto']
        source_all_doc_id = self.TN_doc_id['onto'] + self.TP_doc_id['onto']
        #
        #
        #
        # print(pprefix, 'source_all <-> source TP', self.ave_embedding_similarity(source_all, self.TP['onto']))
        # print(pprefix, 'source_all <-> source TN', self.ave_embedding_similarity(source_all, self.TN['onto']))
        # print(pprefix, 'source_all <-> target TP', self.ave_embedding_similarity(source_all, self.TP['i2b2']))
        # print(pprefix, 'source_all <-> target TN', self.ave_embedding_similarity(source_all, self.TN['i2b2']))
        #
        # self.extrema_distance_examples(self.TP['onto'], source_all, self.TP_starts['onto'], source_all_starts,
        #                                self.TP_ends['onto'], source_all_ends, self.TP_sentences['onto'],
        #                                source_all_sentences, self.TP_span_text['onto'], source_all_spans,
        #                                self.TP_doc_id['onto'], source_all_doc_id, 'TP_onto', 'source_all')
        # self.extrema_distance_examples(self.TN['onto'], source_all, self.TN_starts['onto'], source_all_starts,
        #                                self.TN_ends['onto'], source_all_ends, self.TN_sentences['onto'],
        #                                source_all_sentences, self.TN_span_text['onto'], source_all_spans,
        #                                self.TN_doc_id['onto'], source_all_doc_id, 'TN_onto', 'source_all')
        # self.extrema_distance_examples(self.TP['i2b2'], source_all, self.TP_starts['i2b2'], source_all_starts,
        #                                self.TP_ends['i2b2'], source_all_ends, self.TP_sentences['i2b2'],
        #                                source_all_sentences, self.TP_span_text['i2b2'], source_all_spans,
        #                                self.TP_doc_id['i2b2'], source_all_doc_id, 'TP_i2b2', 'source_all')
        # self.extrema_distance_examples(self.TN['i2b2'], source_all, self.TN_starts['i2b2'], source_all_starts,
        #                                self.TN_ends['i2b2'], source_all_ends, self.TN_sentences['i2b2'],
        #                                source_all_sentences, self.TN_span_text['i2b2'], source_all_spans,
        #                                self.TN_doc_id['i2b2'], source_all_doc_id, 'TN_i2b2', 'source_all')
        #
        #
        # # distance from target embeddings
        target_all = np.concatenate((self.TP['i2b2'], self.TN['i2b2']), axis=0)
        target_all_starts = self.TN_starts['i2b2'] + self.TP_starts['i2b2']
        target_all_ends = self.TN_ends['i2b2'] + self.TP_ends['i2b2']
        target_all_sentences = self.TN_sentences['i2b2'] + self.TP_sentences['i2b2']
        target_all_spans = self.TN_span_text['i2b2'] + self.TP_span_text['i2b2']
        target_all_doc_id = self.TN_doc_id['i2b2'] + self.TP_doc_id['i2b2']

        # print(pprefix, 'target_all <-> source TP', self.ave_embedding_similarity(target_all, self.TP['onto']))
        # print(pprefix, 'target_all <-> source TN', self.ave_embedding_similarity(target_all, self.TN['onto']))
        # print(pprefix, 'target_all <-> target TP', self.ave_embedding_similarity(target_all, self.TP['i2b2']))
        # print(pprefix, 'target_all <-> target TN', self.ave_embedding_similarity(target_all, self.TN['i2b2']))
        #
        # self.extrema_distance_examples(self.TP['onto'], target_all, self.TP_starts['onto'], target_all_starts,
        #                                self.TP_ends['onto'], target_all_ends, self.TP_sentences['onto'],
        #                                target_all_sentences, self.TP_span_text['onto'], target_all_spans,
        #                                self.TP_doc_id['onto'], target_all_doc_id, 'TP_onto', 'target_all')
        # self.extrema_distance_examples(self.TN['onto'], target_all, self.TN_starts['onto'], target_all_starts,
        #                                self.TN_ends['onto'], target_all_ends, self.TN_sentences['onto'],
        #                                target_all_sentences, self.TN_span_text['onto'], target_all_spans,
        #                                self.TN_doc_id['onto'], target_all_doc_id, 'TN_onto', 'target_all')
        # self.extrema_distance_examples(self.TP['i2b2'], target_all, self.TP_starts['i2b2'], target_all_starts,
        #                                self.TP_ends['i2b2'], target_all_ends, self.TP_sentences['i2b2'],
        #                                target_all_sentences, self.TP_span_text['i2b2'], target_all_spans,
        #                                self.TP_doc_id['i2b2'], target_all_doc_id, 'TP_i2b2', 'target_all')
        # self.extrema_distance_examples(self.TN['i2b2'], target_all, self.TN_starts['i2b2'], target_all_starts,
        #                                self.TN_ends['i2b2'], target_all_ends, self.TN_sentences['i2b2'],
        #                                target_all_sentences, self.TN_span_text['i2b2'], target_all_spans,
        #                                self.TN_doc_id['i2b2'], target_all_doc_id, 'TN_i2b2', 'target_all')
        #
        #
        # # internal averages
        print(pprefix, 'source all', self.ave_embedding_similarity(source_all))
        print(pprefix, 'target_all', self.ave_embedding_similarity(target_all))
        # print(pprefix, 'source TP', self.ave_embedding_similarity(self.TP['onto']))
        # print(pprefix, 'source TN', self.ave_embedding_similarity(self.TN['onto']))
        # print(pprefix, 'target TP', self.ave_embedding_similarity(self.TP['i2b2']))
        # print(pprefix, 'target TN', self.ave_embedding_similarity(self.TN['i2b2']))

        print(pprefix, 'source all <-> target_all', self.ave_embedding_similarity(source_all, target_all))
        self.extrema_distance_examples(source_all, target_all, source_all_starts, target_all_starts,
                                       source_all_ends, target_all_ends, source_all_sentences,
                                       target_all_sentences, source_all_spans, target_all_spans,
                                       source_all_doc_id, target_all_doc_id, 'source_all', 'target_all')


        print(pprefix, 'overall', self.ave_embedding_similarity(np.concatenate((source_all, target_all), axis=0)))



    # compute embedding similarity between source and target TP Mentions (system out = gold),
    # TN Mentions (gold mentions that system missed), FP Mentions (system output that does not appear in gold set),
    def load_embeddings(self, gold_clusters, span_embeddings, sentences, coref_predictions, candidate_starts, candidate_ends, doc_id, domain='i2b2'):
        pprefix = '[embedding_sim]'

        # print(pprefix, 'check shapes ...')
        # print(pprefix, 'span_embeddings', span_embeddings.shape)
        # print(pprefix, 'candidate_starts', candidate_starts.shape)
        # print(pprefix, 'candidate_ends', candidate_ends.shape)
        subtokens_mrg = list(itertools.chain.from_iterable(sentences))
        predicted_mentions = list(itertools.chain.from_iterable(coref_predictions))


        # get TN mentions
        for cl_idx, cluster in enumerate(gold_clusters):
            cluster.sort(key=lambda x: x[0])
            # print(pprefix, 'cluster:', cl_idx, cluster)

            for span_idx, span in enumerate(cluster):
                # print('\n')
                span_start = span[0]
                span_end = span[1]

                # try to find span in candidates
                candidate_span_idx = np.intersect1d(np.where(candidate_starts == span_start),
                                                  np.where(candidate_ends == span_end))

                if (len(candidate_span_idx) == 0):
                    print(pprefix, ' '.join(subtokens_mrg[span_start:span_end+1]), ' span mention not in candidate starts/ends')
                    # span_mention_score = np.nan
                else:
                    candidate_idx = candidate_span_idx[0]
                    # print(pprefix, subtokens_mrg[span_start:span_end+1], ' mention score', candidate_mention_scores[span_mention_idx])
                    # span_mention_score = candidate_mention_scores[span_mention_idx]
                    # print(pprefix, 'candidate_span_idx', candidate_span_idx)

                    span_emb = span_embeddings[candidate_idx]
                    span_emb = span_emb.reshape((1, span_emb.shape[0]))
                    # print(pprefix, 'fetched mebedding', span_emb.shape)
                    span_found = (span_start, span_end) in predicted_mentions

                    # save span embedding, save start, end, save span text, save document id
                    if(span_found):
                        # print(pprefix, 'span_found')
                        if (self.TP[domain].size == 0):
                            self.TP[domain] = span_emb
                        else:
                            self.TP[domain] = np.concatenate((self.TP[domain], span_emb), axis=0)
                        self.TP_span_text[domain].append(' '.join(subtokens_mrg[span_start:span_end + 1]))
                        self.TP_doc_id[domain].append(doc_id)
                        self.TP_starts[domain].append(span_start)
                        self.TP_ends[domain].append(span_end)
                        self.TP_sentences[domain].append(subtokens_mrg)
                    else:
                        if (self.TN[domain].size == 0):
                            self.TN[domain] = span_emb
                        else:
                            self.TN[domain] = np.concatenate((self.TN[domain], span_emb), axis=0)
                        self.TN_span_text[domain].append(' '.join(subtokens_mrg[span_start:span_end + 1]))
                        self.TN_doc_id[domain].append(doc_id)
                        self.TN_starts[domain].append(span_start)
                        self.TN_ends[domain].append(span_end)
                        self.TN_sentences[domain].append(subtokens_mrg)

def analysis( candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
             top_antecedents, top_antecedent_scores, predicted_antecedents, sentences, coref_predictions,
             gold_clusters, span_info_list=[], gold_span_info_list=[], write=False):
    pprefix = '[analysis]'
    subtokens_mrg = list(itertools.chain.from_iterable(sentences))
    # span_info_list = []
    if(len(predicted_antecedents) != top_antecedent_scores.shape[0]):
        print('shapes wrong, top_antecedent_scores.shape', top_antecedent_scores.shape, 'len(predicted_antecedents)', len(predicted_antecedents))
        return span_info_list, gold_span_info_list
    # recurse through the predicted antecdents:
    for span_idx, ante_idx in enumerate(predicted_antecedents):

        if (ante_idx == -1):
            # dummy
            if (len(top_antecedent_scores[span_idx]) <51):
                print('error with antecdent score length', top_antecedent_scores[span_idx].shape)
                pairwise_score = top_antecedent_scores[span_idx][-1]
            else:

                pairwise_score = top_antecedent_scores[span_idx][-1]
            ante_score = np.nan
            ante_text = []
            span_start = top_span_starts[span_idx]
            span_end = top_span_ends[span_idx]
            span_text = subtokens_mrg[span_start:span_end + 1]
            span_mention_idx = \
            np.intersect1d(np.where(candidate_starts == span_start), np.where(candidate_ends == span_end))[0]
            span_score = candidate_mention_scores[span_mention_idx]
            ante_start = -1
            ante_end = -1


        else:
            # ante_idx = top_antecedents[span_idx][ante_idx]
            # print('[analysis] np.where(top_antecedents[span_idx]== ante_idx)[0]', np.where(top_antecedents[span_idx]== ante_idx)[0])
            # print(pprefix, 'top_antecedents[span_idx]', top_antecedents[span_idx])
            # print(pprefix, 'ante_idx', ante_idx)
            pairwise_score = top_antecedent_scores[span_idx][np.where(top_antecedents[span_idx] == ante_idx)[0]]
            ante_start = top_span_starts[ante_idx]
            ante_end = top_span_ends[ante_idx]
            # print('[analysis] ante_start', ante_start, 'ante_end', ante_end)

            ante_text = subtokens_mrg[ante_start:ante_end + 1]
            span_start = top_span_starts[span_idx]
            span_end = top_span_ends[span_idx]
            # print(pprefix, 'span_start', span_start, 'span_end', span_end)
            span_text = subtokens_mrg[span_start:span_end + 1]

            # print('[analysis]', span_text, ante_text, antecedent_score)
            # find mention score
            # print(pprefix, 'list(np.where(candidate_starts == span_start))', list(np.where(candidate_starts == span_start)))
            span_mention_idx = \
            np.intersect1d(np.where(candidate_starts == span_start), np.where(candidate_ends == span_end))[0]
            ante_mention_idx = \
            np.intersect1d(np.where(candidate_starts == ante_start), np.where(candidate_ends == ante_end))[0]
            # print(pprefix, 'span mention score', candidate_mention_scores[span_mention_idx], 'ante mention score', candidate_mention_scores[ante_mention_idx])
            span_score = candidate_mention_scores[span_mention_idx]
            ante_score = candidate_mention_scores[ante_mention_idx]
            # print(pprefix, 'span_mention_idx', span_mention_idx)
        pred_span_info = {'ante_score': ante_score,
                          'ante_text': ante_text,
                          'span_text': span_text,
                          'span_score': span_score,
                          'span_start': span_start,
                          'span_end': span_end,
                          'ante_start': ante_start,
                          'ante_end': ante_end,
                          'pairwise_score':pairwise_score}
        span_info_list.append(pred_span_info)
    # print(pprefix, 'coref_predictions', list(itertools.chain.from_iterable(coref_predictions)))
    if(write):
        with open('span_info.csv', 'a', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file,
                                fieldnames=span_info_list[0].keys(),

                                )
            fc.writeheader()
            fc.writerows(span_info_list)


    predicted_mentions = list(itertools.chain.from_iterable(coref_predictions))
    # print(pprefix, 'gold_clusters', gold_clusters)
    # gold_span_info_list = []
    for cl_idx, cluster in enumerate(gold_clusters):
        cluster.sort(key=lambda x: x[0])
        # print(pprefix, 'cluster:', cl_idx, cluster)

        for span_idx, span in enumerate(cluster):
            # print('\n')
            span_start = span[0]
            span_end = span[1]
            span_text = subtokens_mrg[span_start:span_end + 1]
            span_mention_idx = np.intersect1d(np.where(candidate_starts == span_start),
                                              np.where(candidate_ends == span_end))
            span_in_top_mention = False
            ante_in_top_mention = False
            in_top_pairwise = False
            # span mention score
            if (len(span_mention_idx) == 0):
                # print(pprefix, subtokens_mrg[span_start:span_end+1], ' span mention not in candidate starts/ends')
                span_mention_score = np.nan
            else:
                span_mention_idx = span_mention_idx[0]
                # print(pprefix, subtokens_mrg[span_start:span_end+1], ' mention score', candidate_mention_scores[span_mention_idx])
                span_mention_score = candidate_mention_scores[span_mention_idx]
                span_in_top_mention = True
            # get antecdent score
            if (span_idx == 0):
                # print(pprefix,'no antecedent, for the first span in cluster')
                ante_text = []
                ante_mention_score = np.nan
                span_0th = True

            else:
                span_0th = False
                ante_start = cluster[span_idx - 1][0]
                ante_end = cluster[span_idx - 1][1]
                ante_text = subtokens_mrg[ante_start:ante_end + 1]
                ante_mention_idx = np.intersect1d(np.where(candidate_starts == ante_start),
                                                  np.where(candidate_ends == ante_end))
                if (len(ante_mention_idx) == 0):
                    # print(pprefix, subtokens_mrg[ante_start:ante_end + 1], ' antecedent mention not in candidate starts/ends')
                    ante_mention_score = np.nan

                else:
                    # print(pprefix, 'antecdent mention score:', candidate_mention_scores[ante_mention_idx])
                    ante_mention_score = candidate_mention_scores[ante_mention_idx][0]
                    ante_in_top_mention = True
            span_missed = False
            if ((span_start, span_end) not in predicted_mentions):
                span_missed = True
                # print(pprefix, 'MISSED mention', subtokens_mrg[span_start:span_end+1])
            # get the antecdent score
            if (span_idx == 0):
                ante_start = -1
                ante_end = -1
                span_mention_idx = \
                    np.intersect1d(np.where(top_span_starts == span_start), np.where(top_span_ends == span_end))
                # print(pprefix, 'no antecedent', 'span_mention_idx', span_mention_idx)
                if (len(span_mention_idx) == 0):
                    # print(pprefix, 'span not in top_spans')
                    pairwise_score = np.nan

                else:
                    # print(pprefix, 'span_mention_idx', span_mention_idx)
                    # print(pprefix, 'dummy antecdent score', top_antecedent_scores[span_mention_idx[0]][50])
                    if(len(top_antecedent_scores[span_mention_idx[0]]) < 51):
                        pairwise_score = top_antecedent_scores[span_mention_idx[0]][-1]
                        print('pairwise error', (top_antecedent_scores[span_mention_idx[0]]).shape)
                    else:
                        pairwise_score = top_antecedent_scores[span_mention_idx[0]][-1]
                    in_top_pairwise = True

                # antecedent_score = top_antecedent_scores[span_mention_idx][50]
            else:
                ante_start = cluster[span_idx - 1][0]
                ante_end = cluster[span_idx - 1][1]
                # find the "top" idx
                span_mention_idx = \
                    np.intersect1d(np.where(top_span_starts == span_start), np.where(top_span_ends == span_end))
                ante_mention_idx = \
                    np.intersect1d(np.where(top_span_starts == ante_start), np.where(top_span_ends == ante_end))
                if (len(span_mention_idx) == 0 or len(ante_mention_idx) == 0):
                    # print(pprefix, 'antecdent and span pairwise score not found')

                    pairwise_score = np.nan
                else:
                    #
                    # print(pprefix, 'ante_mention_idx', ante_mention_idx[0], 'searching in ', top_antecedents[span_mention_idx])
                    top_ante_idx = np.where(top_antecedents[span_mention_idx] == ante_mention_idx[0])
                    if (len(top_ante_idx) == 0):
                        # print(pprefix, 'antecdent score not found among top scorers for this span')
                        pairwise_score = np.nan
                    else:
                        # print(pprefix, 'pairwise score', top_antecedent_scores[span_mention_idx][top_ante_idx])
                        pairwise_score = top_antecedent_scores[span_mention_idx][top_ante_idx]
                        in_top_pairwise = True
            span_info = {'span_text': span_text,
                         'span_mention_score': span_mention_score,
                         'ante_text': ante_text,
                         'ante_mention_score': ante_mention_score,
                         'span_missed': span_missed,
                         'span_start': span_start,
                         'span_end': span_end,
                         'ante_start': ante_start,
                         'ante_end': ante_end,
                         'pairwise_score': pairwise_score,
                         'span_0th': span_0th,
                         'span_in_top_mention': span_in_top_mention,
                         'ante_in_top_mention': ante_in_top_mention,
                         'in_top_pairwise': in_top_pairwise
                         }
            gold_span_info_list.append(span_info)
            # print(pprefix, 'span_info:', span_info)
            # print(pprefix, 'antecdent idx', ante_mention_idx, 'span_mention_idx', span_mention_idx)
    if(write):
        with open('gold_span_info.csv', 'a', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file,
                                fieldnames=gold_span_info_list[0].keys(),

                                )
            fc.writeheader()
            fc.writerows(gold_span_info_list)
    # for a given predicted antecdent (x,y)
    #   print mention score and x, y
    #   print coref score x, y

    return span_info_list, gold_span_info_list