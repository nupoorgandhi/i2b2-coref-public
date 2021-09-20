#!/bin/bash

source ~/.bashrc
conda activate py36
echo $PATH
#python checkout_devices.py

#GPU=0 python train.py train_spanbert_base_proj ../../config/experiments.conf
#GPU=0 python train.py train_spanbert_base ../../config/experiments.conf
#GPU=0 python evaluate.py spanbert_base ../../config/experiments.conf
