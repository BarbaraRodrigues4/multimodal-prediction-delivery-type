# Training and Evaluation of the Multimodal Models    

  The best model architectures obtained from HPO that were manually replicated are organized as follows:
   - `multimodal_models/MM1/`
   - `multimodal_models/MM2/`
   - `multimodal_models/MM3/`

  Each folder includes:
   - `train_validate.py` for training and internal validation
   - `continued_train.py` for continued training of the previously trained and validated models
  
  **Before running**, update the following variables: 
   - *train_data_path*, *val_data_path*, and *test_data_path* - Set paths to your dataset splits
   - *exp_id* — Set the experiment ID/name
   - *epochs* — Set the number of training epochs
   - *patience* — Set the patience for early stopping (only for train_validate.py)

## Run training with validation:
```bash
python train_validate.py
```

## Run continued training:
```bash
python continued_train.py
```