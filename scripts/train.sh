#!/bin/bash

# For training a single model, run the following command:
python train.py --model_type DeformationClassifier \
                --batch_size 100 \
                --train_data_dir datasets/train_set_N_tension_1250_N_compression_1250_N_rigid_1250/training/ \
                --val_data_dir datasets/train_set_N_tension_1250_N_compression_1250_N_rigid_1250/validation/ \
                --experiment_name train_DC

# For training a subset of the four models, run the following command:
python train.py --train_all \
                --batch_size 10 \
                --model_types TensionNet CompressionNet RigidNet \
                --train_data_dir datasets/train_set_N_tension_1250_N_compression_1250_N_rigid_1250/training/ \
                --val_data_dir datasets/train_set_N_tension_1250_N_compression_1250_N_rigid_1250/validation/ \
                --experiment_name train_TN_CN_RN