for e in lcld_manual
do
  python -m drift_study.generate_config \
      -c config/${e}_generate.yaml \
      -p output_path=config/${e}.yaml
done
