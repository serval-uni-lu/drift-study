#!/bin/bash
for dataset in lcld
do
    for delay in delays_all, delay_none
    do
        for n_input in 100000 300000
        do
            echo "python \
                  -m drift_study.optimize_simulator \
                  -c config/${dataset}_opt.yaml \
                  -c config/${delay}.yaml \
                  -c config/${dataset}_rf.yaml \
                  -c config/rf_optimize.yaml \
                  -p common_runs_params.end_train_idx=${n_input} \
                  -p sub_dir_path=opt_${n_input}_${delay}"
        done
    done
done
