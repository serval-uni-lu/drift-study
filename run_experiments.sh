python -m src.run_simulator -c config/electricity.yaml -c config/electricity_rf.yaml -c config/runs_common.yaml -c config/runs_rf.yaml
python -m src.run_simulator -c config/electricity.yaml -c config/electricity_nn.yaml -c config/runs_common.yaml
python -m src.run_simulator -c config/lcld.yaml -c config/lcld_rf.yaml -c config/runs_common.yaml -c config/runs_rf.yaml
python -m src.run_simulator -c config/lcld.yaml -c config/lcld_nn.yaml -c config/runs_common.yaml
