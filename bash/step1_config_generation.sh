#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of configuration generation
echo "$(date): Starting configuration generation..." | tee -a $LOGFILE
python cfgs/gen_cfgs.py
echo "$(date): Finished configuration generation." | tee -a $LOGFILE
