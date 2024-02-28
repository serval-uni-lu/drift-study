now=$(date +%Y%m%d%H%M%S)
echo $now
zip -r drift-study_${now}.zip drift_study/ config/ jobs/ scripts/ README.md requirements.txt