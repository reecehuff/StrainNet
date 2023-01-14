#!/bin/bash
for MVC in 10 30 50;
do
    for TRIAL in 1 2 3 4 5;
    do
        python apply2experimental.py    --exp_data_dir datasets/experimental/test/${MVC}mvc/trial${TRIAL}/ \
                                        --model_dir models/pretrained/experimental/ \
                                        --sampling_rate 30 \
                                        --save_strains \
                                        --visualize \
                                        --log_dir results/subject1.${MVC}mvc.trial${TRIAL}/
    done
done
