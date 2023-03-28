#!/bin/bash
for dataset in lcld
do
    for delay in F delays_all
    do
        for n_input in 100000 300000
        do
            eval "python \
                  -m drift_study.run_simulator \
                  -c config/logging.yaml \
                  -c config/${dataset}_full.yaml \
                  -c config/${delay}.yaml \
                  -c config/${dataset}_rf_opt.yaml \
                  -c config/${dataset}_runs_opt.yaml \
                  -p common_runs_params.end_train_idx=${n_input} \
                  -p sub_dir_path=opt_${n_input}_${delay}"
        done
    done
done
