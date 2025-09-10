import sys
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import data
import models
import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === CHANGE THESE PATHS AND SETTINGS BEFORE RUNNING ===
trained_model_path = Path("")
train_data_path = ""
test_data_path = ""
experiment = '' #e.g. '1_tv1'
epochs = 50 
# ======================================================

base_dir = Path(__file__).parent.resolve()
model_save_dir = base_dir / "models"
results_save_dir = base_dir / "results"
model_save_dir.mkdir(parents=True, exist_ok=True) 
results_save_dir.mkdir(parents=True, exist_ok=True)
model_new_dir = model_save_dir / f'exp_{experiment}'
model_new_dir.mkdir()
results_new_dir = results_save_dir / f'exp_{experiment}'
results_new_dir.mkdir() 

MODEL_PATH = model_new_dir / "model.pth"
ARCHITECTURE_MODEL = model_new_dir / "architecture_model.txt"
CONFIG_FILE_PATH = model_new_dir / "config_model.json"
HISTORY_FILE_PATH = results_new_dir / "train_history.json"
PLOTS_FILE_PATH = results_new_dir / "training_curves.png"
METRICS_MODEL = results_new_dir / "test_metrics.json"
CM_MODEL = results_new_dir / "conf_matrix.png"
ROC_MODEL = results_new_dir / "roc_curve.png"
PREDI_MODEL = results_new_dir / "pred_probs.csv"

# ---------------------- Config ---------------------- #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = epochs
BATCH_SIZE = 16
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.000373345024672811

with open("columns_config.json", "r") as f:
        col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]
y_col = "Class"

# ---------------------- Data Loading ---------------------- #
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data = train_data.drop('Processo', axis=1)
test_data = test_data.drop('Processo', axis=1)

# ---------------------- Preprocessing & Dataset ---------------------- #
tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
tab_preproc.fit(train_data)

train_dataset = data.MultimodalDataset(train_data, tab_preproc, image_cols, y_col, train=True)
test_dataset = data.MultimodalDataset(test_data, tab_preproc, image_cols, y_col, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------- Model Setup ---------------------- #
num_numerical = len(num_cols)
num_categories = [train_data[col].nunique() + 1 for col in cat_cols]
tabular_token_dim = 192
tabular_hidden_dim = 192

multimodal_model = models.build_multimodal_model(num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim).to(DEVICE)
with open(ARCHITECTURE_MODEL, 'w') as f:
    f.write(str(multimodal_model))
optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()
utils.save_config_file(CONFIG_FILE_PATH, "AdamW", LEARNING_RATE, BATCH_SIZE, EPOCHS)

loaded_state_dict = torch.load(trained_model_path, weights_only=True)
current_state_dict = multimodal_model.state_dict()
compatible_state_dict = {
    k: v for k, v in loaded_state_dict.items()
    if k in current_state_dict and v.shape == current_state_dict[k].shape
}
multimodal_model.load_state_dict(compatible_state_dict, strict=False)

# ---------------------- Training & Validation ---------------------- #
train_results = utils.train_only(multimodal_model, train_loader, optimizer, criterion, DEVICE, MODEL_PATH, EPOCHS)
utils.save_continued_train_metrics(
    history_file_path = HISTORY_FILE_PATH,
    train_losses = train_results["train_losses"],
    train_accuracies = train_results["train_accuracies"],
    elapsed_time = train_results["elapsed_time"]
)
utils.plot_Train_LossAcc(
    PLOTS_FILE_PATH,
    train_results["train_losses"],
    train_results["train_accuracies"],
)

# ---------------------- Test Evaluation ---------------------- #
test_results = utils.evaluate(multimodal_model, test_loader, DEVICE)
utils.save_test_metrics(
    metrics_file_path = METRICS_MODEL,
    balanced_accuracy = test_results["balanced_accuracy"],
    acc = test_results["acc"],
    precision = test_results["precision"],
    recall = test_results["recall"],
    f1 = test_results["f1"],
    roc_auc = test_results["roc_auc"],
    matthews = test_results["matthews"],
    specificity = test_results["specificity"],
    tnr=test_results["tnr"]
)
prob_scores = np.array(test_results["prob_scores"])
utils.save_predictions(
    predictions_file_path=PREDI_MODEL,
    y_true=test_results["targets"],
    y_pred=test_results["predictions"],
    prob_scores=prob_scores
)
class_names = ['Cesarean birth','Vaginal birth']  # label 1, label 0
utils.plot_confusionMatrix(CM_MODEL, test_results["confusion_matrix"], class_names, "Confusion Matrix: Cesarean vs Vaginal Birth")
utils.plot_RocCurve(ROC_MODEL,targets=test_results["targets"],prob_scores=test_results["prob_scores"])