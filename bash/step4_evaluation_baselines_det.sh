#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of evaluation for baseline deterministic model
echo "$(date): Starting evaluation for baseline deterministic model..." | tee -a $LOGFILE
python script/eval_baselines_det.py
echo "$(date): Finished evaluation for baseline deterministic model." | tee -a $LOGFILE
