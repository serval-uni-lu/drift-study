for dataset in  lcld
do
    for metric in mcc
    do
        for batch_size in 20000
        do
            eval "python -m drift_study.visualization.line_graph -c config/${dataset}_manual.yaml -p evaluation_params.metric.name=${metric} -p evaluation_params.batch_size=${batch_size} -p sub_dir_path=manual400 -p output_file=./reports/manual/$dataset/${metric}_${batch_size}.html -p plot_engine=plotly"
        done
    done
done
