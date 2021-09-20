import json
import random
import re
import sys
import numpy as np
DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+) (.*)#end document")
BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")

def sample_docs(data_list, sample_size=3, keys=['00']):
    if (keys == ['00']):
        return random.sample(data_list, min(len(data_list)-1,sample_size))
    data_list_mod = []
    for x in data_list:
        if(x['doc_key'][:2] in keys):
            data_list_mod.append(x)
    print('dat_list_mod', len(data_list_mod))

    if(len(data_list_mod) == 0):
        return []
    return random.sample(data_list_mod,  min(len(data_list_mod)-1,sample_size))


def read_jsonlines(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_conll_lbl(filename, doc_key_set):
    sampled_conll_lines = []
    with open(filename, 'r') as content_file:
        curr_key = ''
        reading_key = False

        for line in content_file:
            row = line.split()
            if len(row) == 0:
                if(reading_key):
                    sampled_conll_lines.append("\n")
            elif(row[0].startswith("#")):
                begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
                if begin_match:
                    doc_key = "{}_{}".format(begin_match.group(1), int(begin_match.group(2)))
                    if(doc_key in doc_key_set):

                        curr_key = doc_key
                        reading_key = True
                        sampled_conll_lines.append(line)
                else:
                    if(reading_key):
                        sampled_conll_lines.append(line)
                        curr_key = ''
                        reading_key = False
            else:
                if(reading_key):
                    sampled_conll_lines.append(line)

    return sampled_conll_lines


def average_chain_len(data, sample_size):
    for key in ["tc", 'bc', 'bn', 'nw', 'wb', 'pt' ]:
        samples = sample_docs(data, sample_size=sample_size, keys=[key])
        average = 0.0
        for sample in samples:
            # print('[len(cl) for cl in sample]', [len(cl) for cl in sample])
            # print('np.array([len(cl) for cl in sample[clusters]]).mean()', np.array([len(cl) for cl in sample['clusters']]).mean())
            average += np.array([len(cl) for cl in sample['clusters']]).mean()
        print('average chain length for', key, average/(1.0*sample_size))




def read_conll(filename):
    with open(filename, 'r') as content_file:
        content = content_file.read()
    return content


if __name__ == "__main__":
    jsonlines_file = sys.argv[1]
    conll_file = sys.argv[2]

    key = 'pt'
    size = 10
    data = read_jsonlines(jsonlines_file)
    average_chain_len(data, size)


    jsonlines_sample = sample_docs(data, sample_size=size, keys=[key])
    doc_key_set = [x['doc_key'] for x in jsonlines_sample]
    conll_sample = read_conll_lbl(conll_file, doc_key_set)

    with open(jsonlines_file[:-len('.jsonlines')]+'_tiny.jsonlines', 'w') as f:
        for item in jsonlines_sample:
            f.write(json.dumps(item) + "\n")
    with open(conll_file + '_tiny', 'w') as f:
        for item in conll_sample:
            f.write(item)


    # doc_match = re.match(DOCUMENT_REGEX, conll_content)
    # print('first group', doc_match.group(1))
    # print('second group', doc_match.group(2))
    # print('third group', doc_match.group(3))


