#!/bin/bash

# # For training all of the models, run the following command:
# python train.py --train_all \
#                 --epochs 2 \
#                 --train_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/training/ \
#                 --val_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/validation/ \
#                 --experiment_name train_all

# # For training a single model, run the following command:
python train.py --model_type DeformationClassifier \
                --epochs 2 \
                --train_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/training/ \
                --val_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/validation/ \
                --experiment_name train_DC

# For training a subset of the four models, run the following command:
python train.py --train_all \
                --model_types TensionNet CompressionNet RigidNet \
                --epochs 2 \
                --train_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/training/ \
                --val_data_dir datasets/train_set_N_tension_5_N_compression_5_N_rigid_5/validation/ \
                --experiment_name train_TN_CN_RN