# Abstract
Recent work has shown fine-tuning neural coreference models can produce strong performance when adapting to different domains. However, at the same time, this can require a large amount of annotated target examples. In this work, we focus on supervised domain adaptation for clinical notes, proposing the use of concept knowledge to more efficiently adapt coreference models to a new domain. We develop methods to improve the span representations via (1) a retrofitting loss to incentivize span representations to satisfy a knowledge-based distance function and (2) a scaffolding loss to guide the recovery of knowledge from the span representation. By integrating these losses, our model is able to improve our baseline precision and F-1 score. In particular, we show that incorporating knowledge with end-to-end coreference models results in better performance on the most challenging, domain-specific spans.

# Citation
```
@inproceedings{gandhi-etal-2021-improving,
    title = "Improving Span Representation for Domain-adapted Coreference Resolution",
    author = "Gandhi, Nupoor  and
      Field, Anjalie  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the Fourth Workshop on Computational Models of Reference, Anaphora and Coreference",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.crac-1.13",
    pages = "121--131",
   
}
```

# Set up data
After obtaining i2b2 and ontonotes data, create a few directories:
- .conll files in conll_dir/
- .txt files in i2b2_data/docs/
- .con files in i2b2_data/concept_data/

## Save UMLS data
### Generate a jobs file:
`python generate_slurm.py --generate_umls --umls_jobs_file umls_dir/umls_jobs.txt --docs_root i2b2_data/docs`
### Run tagger
Modify the file umls.slrm:
- Change the jobs text file to umls_jobs.txt
- Change directory to location of umls_dir
Run the job array

## Save jsonlines files
`python minimize.py cased_config_vocab/vocab.txt i2b2_data/conll_files i2b2_data/jsonlines_dir/ true umls_dir/ i2b2_data/concepts/ `

## Set up experiment config files
Modify experiments.conf

## Train model
### Train on source domain first:
`python train.py model_name config_dir/experiments.conf`
### Set up continued training
`python setup_dir.py log_root/ model_name`
### Train on target domain
`python train.py model_name_tuneTrue config_dir/experiments.conf`

## Evaluate on target domain
`python evaluate.py model_name config_dir/experiments.conf tgt`




