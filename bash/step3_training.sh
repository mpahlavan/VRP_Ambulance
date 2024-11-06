#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of training
echo "$(date): Starting training the model..." | tee -a $LOGFILE
python script/train.py
echo "$(date): Finished training the model." | tee -a $LOGFILE
