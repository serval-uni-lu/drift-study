#!/bin/bash

for dataset in lcld
do
    for delay in delays_half
    do
        eval "sbatch launch.sh python \
            -m drift_study.run_simulator \
            -c config/logging.yaml \
            -c config/${dataset}_complete.yaml \
            -c config/${dataset}_complete_delays_all.yaml \
            -c config/${delay}.yaml \
            -p sub_dir_path=full_${delay} \
            "
    done
    eval "sbatch launch.sh python \
            -m drift_study.run_simulator \
            -c config/logging.yaml \
            -c config/${dataset}_complete.yaml \
            -c config/${dataset}_complete_delays_none.yaml \
            -c config/${delay}.yaml \
            -p sub_dir_path=full_delays_none \
            "
done

