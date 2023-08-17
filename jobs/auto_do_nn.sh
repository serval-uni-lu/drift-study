DATASET_NAME=electricity
MODEL_NAME=torchrln
AUTO_CONFIG=./config/auto/${DATASET_NAME}_${MODEL_NAME}.yaml
MODEL_TYPE=nn

RETRAIN_OPT=true
extra_params=""
if [ "$RETRAIN_OPT" = true ]; then
    extra_params="-p model.optimize=true"
else
    extra_params="-p use_auto_model_tuning=true"
fi

PATH_PREFIX=./data/drift/${DATASET_NAME}/${MODEL_NAME}

# Prepare

python -m drift_study.run.optimize_model -c config/logging.yaml -c ${AUTO_CONFIG}

# # TABLE 1

# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/no_retrain/
# python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/ -p plot_path=${PATH_PREFIX}/baseline.html
# python -m drift_study.run.manual_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/manual/
# python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/ -p plot_path=${PATH_PREFIX}/baseline_manual.html

# # # TABLE 2
# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true -p use_auto_model_tuning=true -p train_window_sizes=17760

# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/periodic/ -c ./config/auto/delays_none.yaml -p train_window_size=17760
# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p model.optimize=true -p schedule_data_path=${PATH_PREFIX}/opt_periodic/ -c ./config/auto/delays_none.yaml -p train_window_size=17760

# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/periodic -p out_path=${PATH_PREFIX}/periodic.csv -p train_window_sizes=17760
# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/opt_periodic -p out_path=${PATH_PREFIX}/opt_periodic.csv -p train_window_sizes=17760

# TABLE 3
# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true ${extra_params}
# python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size ${extra_params}
# python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size_no_delays ${extra_params} -c ./config/auto/delays_none.yaml
# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size -p out_path=${PATH_PREFIX}/train_window_size.csv
# python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size_no_delays -p out_path=${PATH_PREFIX}/train_window_size_no_delays.csv

# DETECTOR

# python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_schedules_opt=true ${extra_params}
# python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_opt/ ${extra_params}
# python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_opt_no_delays/ ${extra_params} -c ./config/auto/delays_none.yaml

# python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params}
# python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt_no_delays/ ${extra_params} -c ./config/auto/delays_none.yaml

# python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params} -c ./config/auto/delays_half.yaml
# python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params} -c ./config/auto/delays_twice.yaml



# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/
# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -c ./config/auto/delays_none.yaml
# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -c ./config/auto/delays_half.yaml
# python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -c ./config/auto/delays_twice.yaml

# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/no_retrain
# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/no_retrain
# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/no_retrain
# python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/no_retrain
# python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/ -p plot_path=${PATH_PREFIX}/detector.html -c config/auto/schedules_${MODEL_TYPE}.yaml
# python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -p plot_path=${PATH_PREFIX}/detector_no_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml
# python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -p plot_path=${PATH_PREFIX}/detector_half_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml
# python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -p plot_path=${PATH_PREFIX}/detector_twice_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml