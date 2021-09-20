# from latent_eval import load_jsonlines
import numpy as np
import json
import re
import os
# generate samples where we have
# given a pair of source and target datasets of size N
# (0, 10, 20, ..., N) target examples and (N, N-10, N-20, ... 0) source examples
# use the same random seed

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

def load_jsonlines(jsonfile_loc):
    with open(jsonfile_loc) as input_file:
        print('[load_jsonlines]', jsonfile_loc)
        num_lines = sum(1 for line in input_file.readlines())
        input_file.seek(0)  # return to first line
        # print('at beginning of file', len(input_file.readlines()))
        lines = input_file.readlines()
    return [json.loads(line) for line in lines]

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

def merge_conll_subset(conll_dir, docs, output_file):
    doc_keys = [x['doc_key'].split('_')[0] for x in docs]
    subset_lines = []
    for doc_key in doc_keys:
        filename = '{}.conll'.format(doc_key)
        with open(os.path.join(conll_dir, filename), "r") as text_file:
            content = text_file.readlines()
        subset_lines.extend(content)

    print('number of lines to write', len(subset_lines))
    with open(output_file, "w") as text_file:
        text_file.writelines(subset_lines)


if __name__ == "__main__":
    np.random.seed(1)
    # fix the dev set for i2b2
    # load training docs
    # for sl in [128, 256, 384, 512]:
    #
    # docs = load_jsonlines('/projects/tir4/users/nmgandhi/coref/data/jsonlines/train.i2b2.english.{}.jsonlines'.format(128))
    # np.random.shuffle(docs)
    # print('len of dataset is', len(docs))
    # dev_set = docs[:50]
    # dev_keys = [x['doc_key'] for x in dev_set]
    # train_set = docs[50:]
    # train_keys = [x['doc_key'] for x in train_set]
    #
    # train_conll = '/projects/tir4/users/nmgandhi/coref/data/conll/train.i2b2.conll'
    # new_train_conll = '/projects/tir4/users/nmgandhi/coref/data/conll/train.i2b2.conll_mod'
    # new_dev_conll = '/projects/tir4/users/nmgandhi/coref/data/conll/dev.i2b2.conll_mod'
    #
    # rewrite_conll_subset([train_keys], [train_conll], new_train_conll)
    # rewrite_conll_subset([dev_keys], [train_conll], new_dev_conll)


    # get the original conll for existing train/dev/test splits (including all metadata)
    train_docs = load_jsonlines('/projects/tir4/users/nmgandhi/coref/data/jsonlines/train.i2b2.english.{}.jsonlines'.format(512))
    dev_docs = load_jsonlines('/projects/tir4/users/nmgandhi/coref/data/jsonlines/dev.i2b2.english.{}.jsonlines'.format(512))
    test_docs = load_jsonlines('/projects/tir4/users/nmgandhi/coref/data/jsonlines/test.i2b2.english.{}.jsonlines'.format(512))
    conll_meta_dir = '/projects/tir4/users/nmgandhi/coref/data/conll/conll_medical_meta'
    merge_conll_subset(conll_meta_dir,
                       train_docs, '/projects/tir4/users/nmgandhi/coref/data/conll/train.i2b2.meta.conll')
    print('finished train')
    merge_conll_subset(conll_meta_dir,
                       dev_docs, '/projects/tir4/users/nmgandhi/coref/data/conll/dev.i2b2.meta.conll')
    print('finished dev')
    merge_conll_subset(conll_meta_dir,
                       test_docs, '/projects/tir4/users/nmgandhi/coref/data/conll/test.i2b2.meta.conll')










