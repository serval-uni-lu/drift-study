DATASET_NAME=electricity
MODEL_NAME=rf_scaled
AUTO_CONFIG=./config/auto/${DATASET_NAME}_${MODEL_NAME}.yaml
MODEL_TYPE=rf

RETRAIN_OPT=true
extra_params=""
if [ "$RETRAIN_OPT" = true ]; then
    extra_params="-p model.optimize=true"
else
    extra_params="-p use_auto_model_tuning=true"
fi

# Prepare

# python -m drift_study.run.optimize_model -c config/logging.yaml -c ${AUTO_CONFIG}

# TABLE 1

# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/no_retrain/
# python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/ -p plot_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/baseline.html
# python -m drift_study.run.manual_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/manual/
# python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/ -p plot_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/baseline_manual.html

# # TABLE 2
# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true -p use_auto_model_tuning=true -p train_window_sizes=17760

# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/periodic/ -c ./config/auto/delays_none.yaml
# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p model.optimize=true -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/opt_periodic/ -c ./config/auto/delays_none.yaml

# TABLE 3
# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true ${extra_params}
# python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size ${extra_params}
# python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size_no_delays ${extra_params} -c ./config/auto/delays_none.yaml
# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size -p out_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size.csv
# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size_no_delays -p out_path=./data/drift/${DATASET_NAME}/${MODEL_NAME}/train_window_size_no_delays.csv



# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_schedules_opt=true ${extra_params}
python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=./data/drift//${DATASET_NAME}/${MODEL_NAME}/detector_opt/ ${extra_params}
