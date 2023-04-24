#!/bin/bash
python eval.py  --val_data_dir datasets/synthetic/04DEF/ \
                --sequential \
                --custom_sampling \
                --log_dir results/synthetic/04DEF/ \
                --visualize \
                --save_strains
