import warnings
import numpy as np
from pathlib import Path
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import confusion_matrix
from torchinfo import summary
from utils import load_data, shuffle_data, save_predictions, save_eval, plot_cm
import json

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    np.random.seed(123) 

    # === CHANGE THESE PATHS AND SETTINGS BEFORE RUNNING ===
    train_data_path= ""
    val_data_path = ""
    test_data_path = ""
    preset = "high_quality_hpo"
    time_limit = 86400 # ~24 hours
    exp_id = "" #e.g. "hpo_cv1" 
    # ======================================================

    base_dir = Path(__file__).parent.resolve()
    model_save_dir = base_dir / "models_hpo"
    results_save_dir = base_dir / "results_hpo"
    model_save_dir.mkdir(parents=True, exist_ok=True) 
    results_save_dir.mkdir(parents=True, exist_ok=True)
    model_new_dir = model_save_dir / f'exp_{exp_id}'
    model_new_dir.mkdir()
    results_new_dir = results_save_dir / f'exp_{exp_id}'
    results_new_dir.mkdir() 

    # --------------------------- Load and shuffle the datasets ----------------------------
    train_data, val_data, test_data = load_data(train_data_path, val_data_path, test_data_path)
    train_data = train_data.drop('Processo', axis=1)
    val_data = val_data.drop('Processo', axis=1)
    train_data, val_data = shuffle_data(train_data, val_data)
    # ------------------------- Set the target label -------------------------
    label_col = 'Class'

    with open("columns_config.json", "r") as f:
        col_config = json.load(f)
    
    num_cols = col_config["num_cols"]
    cat_cols = col_config["cat_cols"]
    image_cols = col_config["image_cols"]
    column_types = {col: "numerical" for col in num_cols}
    column_types.update({col: "categorical" for col in cat_cols})
    column_types.update({col: "image_path" for col in image_cols})

    # ---------------------- HPO & Model Training ----------------------
    dump_path = model_new_dir
    predictor = MultiModalPredictor(label=label_col,path=dump_path)
    predictor.fit(
        train_data=train_data,          
        tuning_data=val_data,                
        presets=preset,
        column_types = column_types,
        time_limit=time_limit, 
        hyperparameter_tune_kwargs = {
            "num_to_keep": 1,    
        }    
    )
    predictor.dump_model(dump_path)

    # -------------------- Model Architecture Information --------------------
    trained_model = predictor._learner._model

    architecture_file_path = results_new_dir / 'architecture_model.txt'
    with open(architecture_file_path, 'w') as f:
        f.write(str(trained_model))
    summary(trained_model)

    # ------------------------ Predictions ------------------------
    predictions = predictor.predict(test_data.drop(columns=label_col))
    pred_path = results_new_dir / 'predictions.csv'
    save_predictions(predictions,test_data,pred_path)

    # ------------------- Evaluate Model Performance -------------------
    scores = predictor.evaluate(test_data, metrics=["roc_auc", "recall", "precision", "accuracy", "f1"])
    print("Test Data Evaluation Results: \n", scores)
    eval_path = results_new_dir / 'test_metrics.csv'
    save_eval(scores, eval_path)

    # ---------------------- Confusion Matrix ----------------------
    y_true = test_data[label_col].values
    cm = confusion_matrix(y_true, predictions,labels=[1, 0])
    cm_path = results_new_dir / 'conf_matrix.png'
    plot_cm(cm, cm_path)