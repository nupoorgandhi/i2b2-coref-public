import sys
import json
import time
import argparse
import fileinput

from trie import * 

#demarcation of trie branch
_end = "_end_"

#CUI codes that are noisy
stopcuis = set(open("data/stopcuis.txt").read().splitlines())

"""
Create dictionary entry for cui match and preferred term
INPUTS:
    cui (str): UMLS CUI code
    term (str): preferred UMLS English name
OUTPUTS:
    (dict): tagged CUI in dict format {"count": (int), "cui": (str), "pref_name": (str)}
"""
def add_match(cui, term):
    match = dict()
    match["cui"] = cui
    match["pref_name"] = term
    match["count"] = 1
    return match

"""
Scans a string and checks if a medical term matches occur
INPUTS:
    trie (dict of dict): character trie where "_end_" demarcates a branch
    word (str): one preprocessed, lowercased document
OUTPUTS:
    (lst of dict): tagged CUIs in dict format {"count": (int), "cui": (str), "pref_name": (str)}
"""
def in_trie(trie, word):
    cuis = dict()
    current_dict = trie

    #current matching key, will add characters
    c = ""

    #checks to make sure the same space is not restarted over and over again (infinite loop)
    same_space = -2

    #holds the last space 
    last_space = -1

    i = 0

    while i < len(word):
        letter = word[i]

        if letter == " ":
            last_space = i

        if letter in current_dict:
            #very first character
            if i == 0:
                current_dict = current_dict[letter]
                c += letter
            else:
                #very first start of new match
                if len(c) == 0:
                    #only possible that new match occurs when previous character is a space
                    if word[i-1] == " ":
                        current_dict = current_dict[letter]
                        c += letter
                    #this is a midword match which we don't want
                    else:
                        current_dict = trie
                        c = ""
                #in the middle of matching something, so continue
                else:
                    current_dict = current_dict[letter]
                    c += letter
        else:
            #we want to restart at the very last space
            current_dict = trie

            if last_space != same_space:
                #forces character scanning to start at last space
                i = last_space
                same_space = last_space     
            else:
                #reset same_space
                space_space = -2

            c = ""

        #end of trie is reached aka match
        if _end in current_dict:
            cui = current_dict[_end]
            #not at end of text and not a midword match
            if i < len(word)-1 and not word[i+1].isalpha():
                #drop 2 letter matches since have been found to be noisy
                if len(cui) > 2:
                    if cui not in stopcuis:
                        if cui in cuis:
                            cuis[cui]["count"] += 1
                        else:
                            cuis[cui] = add_match(cui, c)
                c = ""
        i += 1

    #edge case for very last character fulfilling a match
    if _end in current_dict:
        cui = current_dict[_end]
        #drop 2 letter matches since have been found to be noisy
        if len(cui) > 2:
            if cui not in stopcuis:
                if cui in cuis:
                    cuis[cui]["count"] += 1
                else:
                    cuis[cui] = add_match(cui, c)

    return list(cuis.values())

"""
Dumps the list of list of dictionaries to json format
INPUTS:
    save_path (str): path to save .json
    lst (lst): lst dump of tagged CUIs
"""
def save2json(save_path, lst):
    f = open(save_path, "w")
    json.dump(lst, f, indent=4)
    f.close()

def main(arguments):
    global arg
    # parser = argparse.ArgumentParser(
    #     description = __doc__,
    #     formatter_class = argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-input', help="input path", type=str)
    # parser.add_argument('-output', help="output path", type=str)
    #
    # args = parser.parse_args(arguments)
    input_path = sys.argv[1]
    save_path = sys.argv[2]

    start_time = time.time()

    cui_trie = create_medical_trie()

    print(str(round(time.time() - start_time, 3)), "sec")

    print("Tagging terms...")

    lst_dump = []
    for i, line in enumerate(fileinput.input([input_path])):
        if i % 10000 == 0:
            print("DOCUMENT:", i)
        line = line.strip()
        cuis = in_trie(cui_trie, line)
        lst_dump.append(cuis)

    save2json(save_path, lst_dump)

    print("Done!")
    print(str(round(time.time() - start_time, 3)), "sec")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
