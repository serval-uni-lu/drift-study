for i in 20000 40000
do
    for j in 780000
    do 
        python -m drift_study.visualization.table_periodic \
                -c config/logging.yaml \
                -c config/lcld_periodic.yaml \
                -p evaluation_params.batch_size=${i} \
                -p sub_dir_path=periodic400 \
                -p common_runs_params.test_start_idx=${j}
    done
done