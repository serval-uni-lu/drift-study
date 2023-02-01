LOGLEVEL=ERROR

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_all_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays \
  -p max_pareto=1 \
  -p max_retrain=80

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_all_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays_half \
  -p max_pareto=1 \
  -p max_retrain=80 \
  -p retraining_delay=2W

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_all_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays_twice \
  -p max_pareto=1 \
  -p max_retrain=80 \
  -p retraining_delay=8W

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_no_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=no_delays \
  -p max_pareto=1 \
  -p max_retrain=80

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_label_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=label_delays \
  -p max_pareto=1 \
  -p max_retrain=80

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/lcld_201317_ds_time/rf_lcld/small_retraining_delays6 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=retraining_delays \
  -p max_pareto=1 \
  -p max_retrain=80




sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_all_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays \
  -p max_pareto=1 \
  -p max_retrain=20


sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_all_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays_half \
  -p max_pareto=1 \
  -p max_retrain=20 \
  -p retraining_delay=0.5W

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_all_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=all_delays_twice \
  -p max_pareto=1 \
  -p max_retrain=20 \
  -p retraining_delay=2W


sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_no_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=no_delays \
  -p max_pareto=1 \
  -p max_retrain=20



sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_label_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=label_delays \
  -p max_pareto=1 \
  -p max_retrain=20

sbatch launch.sh python -m drift_study.complete_run \
  -p input_dir=./data/optimizer_results/electricity/rf_classifier/small_retraining_delays4 \
  -p do_run=1 \
  -p performance.n_jobs_optimiser=24 \
  -p sub_dir_path=retraining_delays \
  -p max_pareto=1 \
  -p max_retrain=20




python -m drift_study.complete_run -p input_dir=./data/optimizer_results/electricity/rf_classifier/test -p do_run=false -p performance.n_jobs_optimizer=2 -p sub_dir_path=no_delays -p max_pareto=1
