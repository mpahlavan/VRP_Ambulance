#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of evaluation for baseline stochastic model
echo "$(date): Starting evaluation for baseline stochastic model..." | tee -a $LOGFILE
python script/eval_baselines_stoch.py
echo "$(date): Finished evaluation for baseline stochastic model." | tee -a $LOGFILE
