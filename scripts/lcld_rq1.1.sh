for e in lcld_rq1.1
do
  python -m drift_study.generate_config \
      -c config/${e}_generate.yaml \
      -p output_path=config/${e}.yaml
done

eval "sbatch launch.sh python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_rq1.1.yaml"
