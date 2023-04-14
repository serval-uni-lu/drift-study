for i in {0..49}
do
    eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_periodic_pretrain.yaml -p run_idx=${i}"
done
