#!/bin/bash

# DATASETS=( "electricity" )
DATASETS=("lcld_201317_ds_time" "electricity" )
# MODELS=("torchrln" "vime")
MODELS=("tabnet")
MODELS=("torchrln" "stg" "vime" "tabnet")
adv_training=1

for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        for adv_training in 0 1
        do
            echo "Running $model on $dataset"
            # python -m notebooks.train_model --dataset $dataset --model $model --adv_training $adv_training
            # python -m notebooks.generate_examples --dataset $dataset --model $model --adv_training $adv_training
            python -m notebooks.static_analysis --dataset $dataset --model $model --adv_training $adv_training
        done
    done
done
