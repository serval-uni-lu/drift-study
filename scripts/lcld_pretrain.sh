#for i in 9 19 29 39 49 
#do
#    eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_periodic_pretrain.yaml -p run_idx=${i}"
#done
#    eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_periodic_pretrain.yaml -p run_idx=${i}"
for e in lcld_pretrain
do
    python -m drift_study.generate_config \
           -c config/${e}_generate.yaml \
	   -p output_path=config/${e}.yaml
done

eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_pretrain.yaml"
