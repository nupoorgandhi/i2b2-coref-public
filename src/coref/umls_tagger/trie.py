import re
import pandas as pd

_end = "_end_"

"""
Takes in a medical term dataframe and creates a trie
"""
def make_trie(df):

    print("Creating trie ...")
    root = dict()
    for i, row in df.iterrows():
        current_dict = root
        term = row["term"]
        if type(term) is str:
            term = term.lower()
            term = re.sub(r'[^\w\s]','', term)
            for letter in term:
                current_dict = current_dict.setdefault(letter, dict())
            current_dict = current_dict.setdefault(_end, row["cui"])
    return root

"""
Creates dataframe of medical terms and filters by good categories
"""
def create_medical_trie():
    #Loads .csv containing all medical CUIs
    print("Loading chv.tsv ...")
    chv_df = pd.read_csv("data/chv.tsv", sep="\t", names=["cui","term","chv_pref_name"],\
                        usecols=[0,1,2])
    
    #Loads .csv containing all medical CUIs and their categories
    print("Loading mrsty.csv ...")
    mrsty_df = pd.read_csv("data/mrsty.csv", names=['cui', 'category', 'category_name'], usecols=[0,1,2])

    #Joins CUIs with their categories
    print("Joining on categories ...")
    joined_medical = chv_df.merge(mrsty_df, on="cui")

    #Filters CUIs based on categories specified in 'good' categories
    print("Loading good categories ...")
    goodtypes = open("data/categories.txt").read().splitlines()

    print("Filtering by good categories ...")
    filtered_joined_medical = joined_medical[joined_medical['category_name'].str.contains('|'.join(goodtypes))]
    small = filtered_joined_medical[['cui', 'category_name']].drop_duplicates()
    small.to_csv('data/merged.csv')
    small_merged = small.groupby(['cui'])['category_name'].apply(lambda x: ','.join(x)).reset_index()
    small_merged.to_csv('data/cui_merged.csv')

    print('saved merged')

    cui_trie = make_trie(filtered_joined_medical)

    return cui_trie