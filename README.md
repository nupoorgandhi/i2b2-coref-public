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




# i2b2-coref-public
