now=$(date +%Y%m%d%H%M%S)
echo $now
zip drift-study_patch_${now}.zip $(git diff --name-only --staged)