#!/bin/bash
for DEF in "04DEF" "07DEF" "10DEF" "13DEF" "16DEF"; 
do 
    echo $DEF
    python eval.py  --val_data_dir datasets/synthetic/$DEF/ \
                    --sequential \
                    --custom_sampling \
                    --log_dir results/synthetic/$DEF/ \
                    --visualize \
                    --save_strains
   
done 