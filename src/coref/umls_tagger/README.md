used to process some of the ASD files
Vincent Nguyen

# UMLS Tagger

The **umls_tagger** repo contains code that extracts UMLS (unified medical language system) terms from English text and assigns mapped CUI (concept unique identifier) codes.

## Example

A character trie constructed from mapped English terms and text from an input file is streamed during the tagging of CUI codes. For terms that may have multiple matches such as "abdominal pain", the tagger will pick up the longest term, "abdominal pain" rather than just "abdominal".

The following preprocessed [sentence](https://www.med.unc.edu/medselect/files/sample-3a.pdf):

> most of these episodes have resolved spontaneously without medical care but she has sought medical
> care on several of these occasions in august of 2002 she underwent a laparoscopic cholecystectomy 
> following this operation she continued to have periodic abdominal pain of the same character and at
> the same frequency as what she had been experiencing before her operation

would result in the extracted tags:

```
[
    [
        {
            "pref_name": "operation",
            "cui": "C0543467",
            "count": 2
        },
        {
            "pref_name": "laparoscopic cholecystectomy",
            "cui": "C0162522",
            "count": 1
        },
        {
            "pref_name": "abdominal pain",
            "cui": "C0000737",
            "count": 1
        }
    ]
]
```

## Quick Start

The character trie is created from several data files containing CUI codes, their preferred names, and respective categories.  The exact codes to be included in the trie is filtered by the categories of CUI codes to keep which are stored in **data/categories.txt**.  Exclusion of certain categories allows more pertinent medical terms to be captured. In addition, CUI codes to furthur exclude are stored in **data/stopcuis.txt**.  These codes are ones that have preferred terms that may be noisy (C0080151: said, C0021223: in).

To see the tagger in action, unzip **data/mrsty.csv.zip** and run

```
python umls_tag.py -input data/sample_input.txt -output data/sample_output.txt
```

* `input`: path to file where each document is on a line-separated file
* `output`: path to save in `.json` format where each document is a list of dictionaries where the tagged CUI, preferred name, and count are keys

