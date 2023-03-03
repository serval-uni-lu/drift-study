for i in {0..24}
do
    eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_periodic_no_opt_pretrain.yaml -p run_idx=${i}"
done
