#!/bin/bash

# CONSTANT
DATASET_NAME=electricity
MODEL_NAME=vime
AUTO_CONFIG=./config/auto/${DATASET_NAME}_${MODEL_NAME}.yaml
DATA_ROOT=./data/drift_202402
MODELS_DIR=./data/drift_202402/models
DEFAULT_WINDOW_SIZE=17760
MODEL_TYPE=nn
PATH_PREFIX=${DATA_ROOT}/${DATASET_NAME}/${MODEL_NAME}

RETRAIN_OPT_SEARCH=false

# VARIABLES
RETRAIN_OPT=false
extra_params=""
if [ "$RETRAIN_OPT" = true ]; then
    extra_params="-p model.optimize=true"
else
    extra_params="-p use_auto_model_tuning=true"
fi

extra_params="-c ./config/auto/${DATASET_NAME}_${MODEL_NAME}_training.yaml"
manual_params="-c ./config/auto/${DATASET_NAME}_${MODEL_NAME}_manual.yaml"

# Check if the first argument exists
if [ -z "$1" ]; then
    # If it doesn't exist, set a default value
    stage="all"
else
    # If it exists, use the first argument
    stage="$1"
fi

echo "Running stage: $stage"

# --------------
# | RUN STAGES |
# --------------

# --- STAGE 1 ---

if [ "$stage" = "all" ] || [ "$stage" = 1 ]; then
    # OPTIMIZE MODEL PARAMETERS. This is done only once.
    echo "Running optimize_model"
    eval "python -m drift_study.run.optimize_model -c config/logging.yaml -c ${AUTO_CONFIG} -p data_root=${DATA_ROOT} -p models_dir=${MODELS_DIR}"
    exit
    # BASELINE NO RETRAIN
    echo "Running no_retrain_baseline"
    python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/no_retrain/
    python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/ -p plot_path=${PATH_PREFIX}/baseline.html

    echo "Read the file ${PATH_PREFIX}/baseline.html and check where the model is not performing well."
    echo "Add this in ${AUTO_CONFIG}, manual_retrain."
fi

# --- STAGE 2 ---

if [ "$stage" = "all" ] || [ "$stage" = 2 ]; then
    # BASELINE MANUAL RETRAIN
    echo "Running manual_baseline"
    python -m drift_study.run.manual_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/manual/ ${manual_params}
    python -m drift_study.visualization.line_graph_auto -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/ -p plot_path=${PATH_PREFIX}/baseline_manual.html ${manual_params}
fi

# --- STAGE 3 ---

if [ "$stage" = "all" ] || [ "$stage" = 3 ]; then
    echo "Periodic"
    python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_periodic=true -p use_auto_model_tuning=true
    python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p use_auto_model_tuning=true -p schedule_data_path=${PATH_PREFIX}/periodic/ -c ./config/auto/delays_none.yaml
    python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/periodic -p out_path=${PATH_PREFIX}/periodic.csv -p single_window_size=true

    if [ -z "$RETRAIN_OPT_SEARCH" ]; then
        echo "Periodic with re-train optimization"
        python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} -p model.optimize=true -p schedule_data_path=${PATH_PREFIX}/opt_periodic/ -c ./config/auto/delays_none.yaml
        python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/opt_periodic -p out_path=${PATH_PREFIX}/opt_periodic.csv -p single_window_size=true
    fi
    echo "Add best training parameters to ${AUTO_CONFIG}, training"
fi

# --- STAGE 4 ---
if [ "$stage" = "all" ] || [ "$stage" = 4 ]; then
    echo "Window size"
    python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_window_size=true ${extra_params}
    python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size ${extra_params}
    python -m drift_study.run.train_window_size -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size_no_delays ${extra_params} -c ./config/auto/delays_none.yaml
    python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size -p out_path=${PATH_PREFIX}/train_window_size.csv
    python -m drift_study.visualization.auto_table -c config/logging.yaml -c ${AUTO_CONFIG} -p schedule_data_path=${PATH_PREFIX}/train_window_size_no_delays -p out_path=${PATH_PREFIX}/train_window_size_no_delays.csv
    echo "Add best training parameters to ${AUTO_CONFIG}, training"
fi

# --- STAGE 5 ---
if [ "$stage" = "all" ] || [ "$stage" = 5 ]; then
    echo "Detectors"

    #    DETECTOR

    # Optimization

    python -m drift_study.run.pretrain_models -c config/logging.yaml -c ${AUTO_CONFIG} -p pretrain_schedules_opt=true ${extra_params}
    python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_opt/ ${extra_params}
    python -m drift_study.run.optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_opt_no_delays/ ${extra_params} -c ./config/auto/delays_none.yaml

    # Evaluation
    python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params}
    python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt_no_delays/ ${extra_params} -c ./config/auto/delays_none.yaml

    python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params} -c ./config/auto/delays_half.yaml
    python -m drift_study.run.use_optimize_detector -c config/logging.yaml -c ${AUTO_CONFIG} -c config/auto/schedules_${MODEL_TYPE}.yaml -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -p detector_optimization_path=${PATH_PREFIX}/detector_opt/ ${extra_params} -c ./config/auto/delays_twice.yaml


    # Copy of the baselines
    python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/
    python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -c ./config/auto/delays_none.yaml
    python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -c ./config/auto/delays_half.yaml
    python -m drift_study.run.periodic_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -c ./config/auto/delays_twice.yaml

    python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/no_retrain
    python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/no_retrain
    python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/no_retrain
    python -m drift_study.run.no_retrain_baseline -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/no_retrain

    # Visualization
    python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector/ -p plot_path=${PATH_PREFIX}/detector.html -c config/auto/schedules_${MODEL_TYPE}.yaml
    python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_no_delays/ -p plot_path=${PATH_PREFIX}/detector_no_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml
    python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_half_delays/ -p plot_path=${PATH_PREFIX}/detector_half_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml
    python -m drift_study.visualization.plot_auto -c config/logging.yaml -c ${AUTO_CONFIG} ${extra_params} -p schedule_data_path=${PATH_PREFIX}/detector_twice_delays/ -p plot_path=${PATH_PREFIX}/detector_twice_delays.html -c config/auto/schedules_${MODEL_TYPE}.yaml

fi
