#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of data generation
echo "$(date): Starting validation data generation..." | tee -a $LOGFILE
python script/gen_val_data.py
echo "$(date): Finished validation data generation." | tee -a $LOGFILE
