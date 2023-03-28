#!/bin/bash
for dataset in lcld
do
    for delay in delays_all
    do
        eval "python \
              -m drift_study.run_simulator \
              -c config/logging.yaml \
              -c config/${dataset}_full.yaml \
              -c config/${delay}.yaml \
              -c config/${dataset}_rf_opt.yaml \
              -c config/${dataset}_runs_opt.yaml \
              -p sub_dir_path=full_${delay}"
    done
done
