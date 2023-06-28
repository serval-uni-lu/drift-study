#!/bin/bash
for dataset in lcld
do
    for delay in delays_all
    do
        for n_input in 300000
        do
            eval "python \
                  -m drift_study.optimize_simulator \
                  -c config/logging.yaml \
                  -c config/${dataset}_opt.yaml \
                  -c config/${delay}.yaml \
                  -c config/${dataset}_rf_opt.yaml \
                  -c config/${dataset}_rf_complete.yaml \
                  -p common_runs_params.end_train_idx=${n_input} \
                  -p sub_dir_path=opt_${n_input}_${delay}_all_test"
        done
    done
done
