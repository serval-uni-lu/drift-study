#!/bin/bash
for dataset in lcld
do
    for delay in delays_none
    do
        eval "python \
              -m drift_study.optimize_simulator \
              -c config/logging.yaml \
              -c config/${dataset}_opt.yaml \
              -c config/rf_optimize.yaml \
              -c config/${delay}.yaml \
              -p sub_dir_path=opt400_${delay}"
    done
done
