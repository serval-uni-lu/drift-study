#!/bin/bash
for dataset in lcld
do
    for delay in delays_all, delay_none
    do
        echo "python
              -m drift_study.optimize_simulator
              -c config/${dataset}_opt.yaml
              -c config/${delay}.yaml
              -c config/${dataset}_rf.yaml
              -c config/rf_optimize.yaml
              -p sub_dir_path=opt_delays_all
              -p models_dir=./models"
    done
done
