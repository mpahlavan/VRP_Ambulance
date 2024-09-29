#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of evaluations for learned models
echo "$(date): Starting evaluation for learned deterministic model..." | tee -a $LOGFILE
python script/eval_learned_det.py
echo "$(date): Finished evaluation for learned deterministic model." | tee -a $LOGFILE

echo "$(date): Starting evaluation for learned dynamic model..." | tee -a $LOGFILE
python script/eval_learned_dyn.py
echo "$(date): Finished evaluation for learned dynamic model." | tee -a $LOGFILE

echo "$(date): Starting evaluation for learned stochastic model..." | tee -a $LOGFILE
python script/eval_learned_stoch.py
echo "$(date): Finished evaluation for learned stochastic model." | tee -a $LOGFILE
