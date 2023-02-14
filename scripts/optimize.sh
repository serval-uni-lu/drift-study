sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/lcld.yaml \
  -c config/delays_all.yaml \
  -c config/lcld_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/lcld_small.yaml \
  -p sub_dir_path=small_all_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models5

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/lcld.yaml \
  -c config/delays_none.yaml \
  -c config/lcld_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/lcld_small.yaml \
  -p sub_dir_path=small_no_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models6

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/lcld.yaml \
  -c config/delays_label.yaml \
  -c config/lcld_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/lcld_small.yaml \
  -p sub_dir_path=small_label_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models7

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/lcld.yaml \
  -c config/delays_retraining.yaml \
  -c config/lcld_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/lcld_small.yaml \
  -p sub_dir_path=small_retraining_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models8


# ------------ ELECTRICITY ---------------

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/electricity.yaml \
  -c config/delays_all.yaml \
  -c config/electricity_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/electricity_small.yaml \
  -p sub_dir_path=small_all_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models5

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/electricity.yaml \
  -c config/delays_none.yaml \
  -c config/electricity_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/electricity_small.yaml \
  -p sub_dir_path=small_no_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models6

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/electricity.yaml \
  -c config/delays_label.yaml \
  -c config/electricity_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/electricity_small.yaml \
  -p sub_dir_path=small_label_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models7

sbatch launch.sh python -m drift_study.optimize_simulator \
  -c config/electricity.yaml \
  -c config/delays_retraining.yaml \
  -c config/electricity_rf.yaml \
  -c config/rf_optimize.yaml \
  -c config/electricity_small.yaml \
  -p sub_dir_path=small_retraining_delays \
  -p models_dir=/scratch/users/tsimonetto/drift-study/models8
