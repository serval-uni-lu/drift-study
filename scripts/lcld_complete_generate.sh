#!/bin/bash
for e in delays_none delays_all
do 
    python -m drift_study.complete_run \
            -p input_dir=data/optimizer_results/lcld_201317_ds_time/rf_lcld_400/opt400_${e} \
            -p output_file=config/lcld_complete_${e}.yaml \
            -p max_pareto=1
done