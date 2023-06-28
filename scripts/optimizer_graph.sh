#!/bin/bash
python -m drift_study.visualization.plot_ml_ntrain -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/opt_300000_delays_all_all_test/ -p output_file=reports/optimizer/lcld.html -p plot_engine=plotly
