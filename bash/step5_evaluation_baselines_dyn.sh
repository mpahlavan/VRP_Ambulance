#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of evaluation for baseline dynamic model
echo "$(date): Starting evaluation for baseline dynamic model..." | tee -a $LOGFILE
python script/eval_baselines_dyn.py
echo "$(date): Finished evaluation for baseline dynamic model." | tee -a $LOGFILE
