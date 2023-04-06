#!/bin/bash

# For resuming training of a model, run the following command:
python train.py --resume models/train_DC/ \
                --model_type DeformationClassifier \
                --epochs 2 \
                --train_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/training/ \
                --val_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/validation/ 