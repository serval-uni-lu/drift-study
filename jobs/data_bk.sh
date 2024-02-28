now=$(date +%Y-%m-%d--%H%-M-%S)
echo $now
zip -r drift_study_data_bk_${now}.zip ./data