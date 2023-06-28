for e in lcld_periodic lcld_periodic_pretrain
do
  python -m drift_study.generate_config \
      -c config/${e}_generate.yaml \
      -p output_path=config/${e}.yaml
done
