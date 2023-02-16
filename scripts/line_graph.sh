for dataset in lcld_prerun_40 lcld_prerun_40
do
    for metric in mcc accuracy balanced_accuracy
    do
        for batch_size in 20000
        do
            eval "python -m drift_study.line_graph -c config/$dataset.yaml -p evaluation_params.metric.name=${metric} -p evaluation_params.batch_size=${batch_size} -p sub_dir_path=prerun -p output_file=./reports/prerun/$dataset/prerun_${metric}_$batch_size.html -p plot_engine=plotly"
        done
    done
done


dataset=lcld_prerun
for metric in mcc accuracy balanced_accuracy
do
    for batch_size in 10000 20000
    do
        eval "python -m drift_study.line_graph -c config/$dataset.yaml -p evaluation_params.metric.name=${metric} -p evaluation_params.batch_size=${batch_size} -p sub_dir_path=prerun -p output_file=./reports/prerun/$dataset/prerun_${metric}_$batch_size.html -p plot_engine=plotly"
    done
done


dataset=electricity_prerun
for metric in mcc accuracy balanced_accuracy
do
    for batch_size in 1000
    do
        eval "python -m drift_study.line_graph -c config/$dataset.yaml -p evaluation_params.metric.name=${metric} -p evaluation_params.batch_size=${batch_size} -p sub_dir_path=prerun -p output_file=./reports/prerun/$dataset/prerun_${metric}_$batch_size.html -p plot_engine=plotly"
    done
done
