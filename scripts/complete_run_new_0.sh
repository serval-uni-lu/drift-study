#!/bin/bash
python -m drift_study.complete_run -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/opt_300000_delays_all_all_test -p output_file=./config/lcld_runs_opt.yaml -p do_run=False -p sub_dir_path=test -p max_pareto=1
