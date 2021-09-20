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
import sys
import torch
from debug_util import hdf5_indices, load_pickle
from emb_master import EMB_TRAIN_FILES, EMB_DEV_FILES, EMB_TEST_FILES, EMB_ROOT, SAMPLE_DIR


def find_beginning(seq, lines):
    indices = [i for i, x in enumerate(lines) if x == seq]
    if(len(indices) == 0):
        print('not found', seq, lines[:5])
    return indices[0]
def find_ending(begin_idx, lines):
    for i in range(begin_idx, len(lines)):
        if(lines[i] == '#end document\n'):
            return i
    print('uneneding doc')
    return -1


def rewrite_conll_subset(doc_keys, conll_files, output_file):
    subset_lines = []
    for idx, conll_file in enumerate(conll_files):
        with open(conll_file, "r") as text_file:
            content = text_file.readlines()
        for doc_key in doc_keys[idx]:
            # for ontonotes extract the key and part
            DOCKEY_ONTO_REGEX = re.compile(r"(.*)_(\d+)")

            dockey_match = re.match(DOCKEY_ONTO_REGEX, doc_key)
            doc_id, part = dockey_match.group(1), f'{int(dockey_match.group(2)):03}'
            if 'i2b2' in conll_file:
                begin_seq = "#begin document "+doc_id+'\n'
            else:
                begin_seq = "#begin document (" + doc_id + "); part " + part + '\n'
            begin_idx = find_beginning(begin_seq, content)
            end_idx = find_ending(begin_idx, content)
            # print(begin_idx, end_idx)
            subset_lines.extend(content[begin_idx:end_idx + 1])

    with open(output_file, "w") as text_file:
        text_file.writelines(subset_lines)

# maybe switch to yaml since .conf does not support nesting
def get_input_filename(input_dataset, config, data_partition):
    src_path = 'src_' + data_partition + '_path'
    src_conll_path = 'src_' + data_partition + '_conll_path'
    tgt_path = 'tgt_' + data_partition + '_path'
    tgt_conll_path = 'tgt_' + data_partition + '_conll_path'
    if input_dataset == 'src':
        return [config[src_path]], [config[src_conll_path]]
    elif input_dataset == 'tgt':
        return [config[tgt_path]], [config[tgt_conll_path]]
    else:
        return [config[src_path], config[tgt_path]], [config[src_conll_path], config[tgt_conll_path]]




if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    model_name = sys.argv[1]
    input_dataset = sys.argv[3]
    max_examples = int(sys.argv[4])
    num_examples = [max_examples]
    data_partition = sys.argv[5]



    projection_labels = []
    projection_paths = []


    # load the doc_keys in the .h5 files from emb_master
    if data_partition == 'train':
        emb_filename = EMB_TRAIN_FILES[max_examples][input_dataset]
    elif data_partition == 'dev':
        emb_filename = EMB_DEV_FILES[input_dataset]
    elif data_partition == 'test':
        emb_filename = EMB_TEST_FILES[input_dataset]


    # valid_doc_keys = hdf5_indices(os.path.join(EMB_ROOT, emb_filename))
    # used for the concept stuff
    # can find the file using the maxex and then take the keys of the file
    # then, check that the doc_key is within this set


    # allow user to define number of examples (current default is 100)

    # # ratio of source to target examples (e.g. .2)
    # if(input_dataset == 'src_tgt_all' or input_dataset == 'tgt_src_all'):
    #     num_examples = [max_examples, max_examples]
    #     src_tgt_ratio = float(sys.argv[4])
    #
    #
    # elif(input_dataset == 'src_tgt' or input_dataset == 'tgt_src'):
    #     # print('round(float(sys.argv[4]))', (float(sys.argv[4])))
    #     src_tgt_ratio = float(sys.argv[4])
    #     num_src = int(max_examples *1.0 * src_tgt_ratio)
    #     num_examples = [num_src, max_examples - num_src]
    #     # should be 0 if you want all target, 1 for all source
    # else:
    #     src_tgt_ratio = 0
    #     num_examples = [max_examples] # just one dataset
    # print('num_examples', num_examples, len(sys.argv))



    # Input file in .jsonlines/conll format.
    input_filenames, input_conll = get_input_filename(input_dataset, config, data_partition)



    output_dir = os.path.join(config['emb_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_id = 'partition=' + data_partition + '_dataset='+input_dataset + '_maxex=' + str(max_examples)
    output_file = 'model='+model_name + '_' + output_id + '.h5'
    out_filename = os.path.join(output_dir, output_file)
    if os.path.exists(out_filename):
        os.remove(out_filename)

    # Span embeddings will be written to this file in .h5 format.

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()


    # one dataset


    with tf.Session() as session:
        model.restore(session)
        # doc_keys = []
        doc_keys = {}
        candidate_span_list = []
        cl_id_count = 0
        for idx, input_filename in enumerate(input_filenames):
            with open(input_filename) as input_file:
                print('[input_file]', input_filename)

                doc_keys[idx] = []

                write_count = 0
                num_lines = sum(1 for line in input_file.readlines())
                input_file.seek(0)  # return to first line
                # print('at beginning of file', len(input_file.readlines()))
                lines = input_file.readlines()
                if max_examples < len(lines):
                    random.shuffle(lines)
                    print('Shuffling')
                else:
                    print('not shuffling')




                # # for dev target set:
                # if os.path.exists(os.path.join(SAMPLE_DIR, 'sample_dict.pickle')):
                #     sample_dict = load_pickle(os.path.join(SAMPLE_DIR, 'sample_dict.pickle'))
                #     print('sample_keys', sample_dict.keys())
                #     sample_id = '{}-{}-{}'.format(0,100,0)
                #     dev_keys = sample_dict[sample_id]['src_train_keys']
                #     valid_doc_keys = dev_keys
                #     print('FOND TRAIN KEYS:', len(valid_doc_keys), valid_doc_keys[:5])



                for example_num, line in enumerate(tqdm(lines)):
                    # print('in loop:', example_num)
                    example = json.loads(line)
                    # if example['doc_key'] not in valid_doc_keys:
                    #     continue

                    doc_keys[idx].append(example['doc_key'])

                    # doc_keys.append(example['doc_key'])

                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                    candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)

                    # print('candidate_span_emb:', candidate_span_emb.shape)
                    # print('candidate_starts:', candidate_starts.shape)
                    # print('length of the ')
                    # if sys.argv[7] == 'one_doc':
                    candidate_span_info = span_util.get_emb(example['clusters'], candidate_span_emb,
                                                            candidate_starts,
                                                            candidate_ends,
                                                            example['token_ids'],
                                                            project_paths=projection_paths,
                                                            balanced=config['balanced'], info=True,
                                                            cl_id_count = cl_id_count)

                    sys.stdout.flush()
                    with h5py.File(out_filename, 'a') as hf:
                        # print('writing to', out_filename)


                        span_emb = candidate_span_info.eval()
                        # print('span_emb shape', span_emb.shape)
                        if input_dataset == 'src':
                            hf.create_dataset(str(example_num), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])),
                                              compression="gzip", compression_opts=0, shuffle=False, chunks=True)
                            # print('span_emb', span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])).shape, type(span_emb))
                        else:
                            # print('doc_key', example['doc_key'])
                            hf.create_dataset(str(example['doc_key']), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])),
                                              compression="gzip", compression_opts=0, shuffle=False, chunks=True)
                    cl_id_count += len(example['clusters'])
                    # candidate_span_list.extend([(example_num, candidate_span_info)])
                    # print('added new span_info', len(candidate_span_list))

                    # if example_num % 1 == 0:
                    #     print('Writing files: {}'.format(out_filename))
                    #     sys.stdout.flush()
                    #     with h5py.File(out_filename, 'a') as hf:
                    #
                    #         # save embeddings
                    #         for i, span_emb in candidate_span_list:
                    #             span_emb = span_emb.eval()
                    #             # print('span_emb shape', span_emb.shape)
                    #
                    #             # print('span_emb', span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])).shape, type(span_emb))
                    #             hf.create_dataset(str(i), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])),
                    #                               compression="gzip", compression_opts=0, shuffle=False, chunks=True)
                    #     candidate_span_list = []



                    if (example_num) > num_examples[idx] or (example_num+1) == num_lines:
                        break
        # print('Writing files: {}'.format(out_filename))
        # sys.stdout.flush()
        # # if sys.argv[7] == 'one_doc':
        # #     print('candidate_span_list first element', candidate_span_list[0], candidate_span_list[0].shape)
        # #     # convert candidate_span_list to just one element
        # #     candidate_span_list = [tf.concat(candidate_span_list, 0)]
        # #     print('AFTER candidate_span_list first element', candidate_span_list[0], candidate_span_list[0].shape)
        # #
        #
        #
        # with h5py.File(out_filename, 'w') as hf:
        #     for i, span_emb in enumerate(candidate_span_list):
        #         span_emb = span_emb.eval()
        #         # print('span_emb shape', span_emb.shape)
        #
        #         # print('span_emb', span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])).shape, type(span_emb))
        #         hf.create_dataset(str(i), data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])), compression="gzip", compression_opts=0, shuffle=False, chunks=True)
        #         # hf.create_dataset(doc_keys[i], data=span_emb.reshape((1, span_emb.shape[0], span_emb.shape[1])), compression="gzip", compression_opts=0, shuffle=False, chunks=True)
        #     # hf.create_dataset("span_representations", data=parent_child_reps, compression="gzip", compression_opts=0, shuffle=True, chunks=True)
        #
        #
        #
        # # print('doc_keys', doc_keys)
        # # print('input_conll', input_conll)
        # # rewrite_conll_subset(doc_keys, input_conll, os.path.join(config['conll_dir'], 'model='+model_name + '_' + output_id + '.extracted.conll'))
