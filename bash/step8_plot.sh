#!/bin/bash

# Define the log file
LOGFILE="run_logfile.txt"

# Log start of visualizations 
echo "$(date): Starting generate visualization(plot learn curve)..." | tee -a $LOGFILE
python script/plot_learn_curves.py
echo "$(date): Finished generate visualization(plot learn curve)..." | tee -a $LOGFILE

# Log start of
echo "$(date): Starting generate visualization(plot routs)..." | tee -a $LOGFILE
python script/plot_routes.py
echo "$(date): Finished generate visualizations(plot routs)..." | tee -a $LOGFILE

# Log start of
echo "$(date): Starting result to text..." | tee -a $LOGFILE
python script/results_to_tex.py
echo "$(date): Finished result to text..." | tee -a $LOGFILE

# Log start of
echo "$(date): Starting routs to text..." | tee -a $LOGFILE
python script/routes_to_tex.py
echo "$(date): Finished routs to text..." | tee -a $LOGFILE