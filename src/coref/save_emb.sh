#!/bin/bash


source ~/.bashrc
conda activate py36
echo $PATH
python checkout_devices.py


python extract_span.py spanbert_base ../../config/span_emb.conf src 200 train


