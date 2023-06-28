for e in lcld_rq1.3
do
  # python -m drift_study.generate_config \
  #     -c config/${e}_generate.yaml \
  #     -p output_path=config/${e}.yaml
  # python -m drift_study.run_simulator -c config/logging.yaml -c config/lcld_rq1.3.yaml
  python -m drift_study.visualization.table -c config/logging.yaml -c config/lcld_rq1.3.yaml
done