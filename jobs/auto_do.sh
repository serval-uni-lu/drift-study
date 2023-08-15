AUTO_CONFIG=./config/lcld_rln.yaml; python -m drift_study.run.optimize_model -c config/logging.yaml -c ${AUTO_CONFIG}


python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true -p use_auto_model_tuning=true
python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true -p use_auto_model_tuning=true
python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/lcld_201317_ds_time/rf_scaled/detector_opt/
