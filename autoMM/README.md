# Hyperparameter Optimization (HPO) with AutoMM

  The HPO process is handled by the `model_hpo_search.py` script.

  **Before running**, update the following variables: 
   - *train_data_path*, *val_data_path*, and *test_data_path* - Set paths to your dataset splits
   - *exp_id* — Set the experiment ID/name

   The code is set to run for 24 hours, with the high_quality_hpo preset. If you want to change the time limit or the preset, feel free to also change those variables.

## Run the script:
```bash
python model_hpo_search.py
```