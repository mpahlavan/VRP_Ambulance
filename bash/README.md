### How to Execute
Now, instead of running everything in a single script, you can execute each script one by one:

1. Run Step 1:
   ```bash
   bash bash/step1_config_generation.sh
   ```

2. Run Step 2:
   ```bash
   bash bash/step2_data_generation.sh
   ```

3. Run Step 3:
   ```bash
   bash bash/step3_training.sh
   ```

4. Run Step 4:
   ```bash
   bash bash/step4_evaluation_baselines_det.sh
   ```

5. Run Step 5:
   ```bash
   bash bash/step5_evaluation_baselines_dyn.sh
   ```

6. Run Step 6:
   ```bash
   bash bash/step6_evaluation_baselines_stoch.sh
   ```

7. Run Step 7:
   ```bash
   bash bash/step7_evaluation_learned_models.sh
   ```

Each of these steps will log the process in the same `run_logfile.txt` file, and you can monitor the logs for each separate part.