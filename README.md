# Drift retraining schedules replication package

## Introduction

This package contains all the necessary data and scripts to replicate the experiments of our research paper Predictive Maintenance of Industrial Machine Learning Systems
in Production" on public data.

## Create the environment

We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) and Python 3.8.10 on a Linux system.

```
conda create -n drift-study python=3.8.10
conda activate drift-study
pip install -r requirements.txt
```

## File structure

- config: configurations files used to prepare and run the experiments
- data: the source data set and the results generated by our experiments
  - optmizer_results: the result of the evaluation of the retraining schedules
- drift_study: python modules to run the experiments
- models: pre-trained models
- reports: the final reports of experiment as tables(.csv) and figures (.html)
- scripts: the script to run an experiment

## Usage

Run the following command to run an experiment $rq among [rq1, rq2, rq3, rq4].

```
conda activate drift-study
chmod +x ./script/lcld_${rq}.sh
./script/lcld_${rq}.sh
```

Note that by default, to limit computational costs, only the final reports generation is executed from pre-computed results.
Feel free to uncomment python commands in `./script/lcld_${rq}.sh` to run the complete experiments.

The reports are found in `./reports/lcld_201317_ds_time/$rq.[csv|html]`

## New usage

From a configuration as yaml file

```yaml
---
name: lcld_201317_ds_time
dataset: lcld_201317_ds_time
metric:
  - name: mcc
models:
  - name: rf
detectors:
  - name: divergence
first_train_window_size: -1
train_window_size: [200000]
periods: [20000, 50000, 100000, 200000]
test_start_idx: 400000
training_step_size: 5000
common_detectors_params:
  - batch_size: 5000
delays:
  drift: 0
  label: 10days
  retraining: 4W
optimization_iter:
  model: 100
  detector: 100
```

Do

1. [x] Optimize model hyperparameters.
1. [x] Nothing baseline
1. Periodic retraining
   1. [x] Pretrain model (periodic retraining).
   1. Periodic retraining baseline.
   1. Show results of periodic retraining baseline (Table + trace).
1. Optimization/warm-up of drift detectors.
   1. [x] Pretrain models. (Optimization phase).
   1. [x] Optimization/warm-up of drift detectors.
1. Drift scheduler
   1. [x] Pretrain models. (Main phase).
   1. Run Schedules.
   1. Show results of drift detectors (Graph + Trace).

Output files structure:

```
data/drift/DATASET/MODEL/
  - model_opt/
    - optuna_sampler.joblib
    - optuna_study.db
    - best_params.json
    - best_metrics.json
    - model_params_iter_I.json
    - metrics_iter_I_fold_J.json
    - metrics_iter_I.json
  - schedules/
    - SCHEDULENAME/metrics.json
    - SCHEDULENAME/models_idx.parquet
    - SCHEDULENAME/detector_metrics.parquet
    - SCHEDULENAME/config.json
    - SCHEDULENAME/preds.parquet
  - detector_opt/
    - SCHEDULENAME/
      - best_params.json
      - best_metrics.json
      - optuna_sampler.joblib
      - optuna_study.db
      - schedules/ # same as above
models/DATASET/MODEL/
  - MODELNAME_STARTIDX_ENDIDX.model
  - MODELNAME_name_STARTIDX_ENDIDX_cache.parquet
```
