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
train_data_path = ""
val_data_path = ""
test_data_path = ""
experiment = '' #e.g. '1_cv1'
epochs = 100  
patience = 15
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
best_save_dir = results_new_dir / "best_model"
best_save_dir.mkdir()
final_save_dir = results_new_dir / "final_model" 
final_save_dir.mkdir()

BEST_MODEL_PATH = model_new_dir / "best_model.pth"
FINAL_MODEL_PATH = model_new_dir / "final_model.pth"
ARCHITECTURE_MODEL = model_new_dir / "architecture_model.txt"
CONFIG_FILE_PATH = model_new_dir / "config_model.json"
HISTORY_FILE_PATH = results_new_dir / "train_history.json"
PLOTS_FILE_PATH = results_new_dir / "training_curves.png"

METRICS_BEST_MODEL = best_save_dir / "test_metrics.json"
PREDI_BEST_MODEL = best_save_dir / "pred_probs.csv"
CM_BEST_MODEL = best_save_dir / "conf_matrix.png"
ROC_BEST_MODEL = best_save_dir / "roc_curve.png"
METRICS_FINAL_MODEL = final_save_dir / "test_metrics.json"
CM_FINAL_MODEL = final_save_dir / "conf_matrix.png"
ROC_FINAL_MODEL = final_save_dir / "roc_curve.png"
PREDI_FINAL_MODEL = final_save_dir / "pred_probs.csv"

# ---------------------- Config ---------------------- #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = epochs
BATCH_SIZE = 16
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.000373345024672811
PATIENCE = patience

with open("columns_config.json", "r") as f:
        col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]
y_col = "Class"

# ---------------------- Data Loading ---------------------- #
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

train_data = train_data.drop('Processo', axis=1)
val_data = val_data.drop('Processo', axis=1)
test_data = test_data.drop('Processo', axis=1)

# ---------------------- Preprocessing & Dataset ---------------------- #
tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
tab_preproc.fit(train_data)

train_dataset = data.MultimodalDataset(train_data, tab_preproc, image_cols, y_col, train=True)
val_dataset = data.MultimodalDataset(val_data, tab_preproc, image_cols, y_col, train=False)
test_dataset = data.MultimodalDataset(test_data, tab_preproc, image_cols, y_col, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

# ---------------------- Training & Validation ---------------------- #
train_results = utils.train_and_validate(multimodal_model, train_loader, val_loader, optimizer, criterion, DEVICE, BEST_MODEL_PATH, FINAL_MODEL_PATH, EPOCHS, PATIENCE)
utils.save_train_metrics(
    history_file_path = HISTORY_FILE_PATH,
    train_losses = train_results["train_losses"],
    val_losses = train_results["val_losses"],
    train_accuracies = train_results["train_accuracies"],
    val_accuracies = train_results["val_accuracies"],
    best_epoch = train_results["best_epoch"],
    elapsed_time = train_results["elapsed_time"]
)
utils.plot_TrainVal_LossAcc(
    PLOTS_FILE_PATH,
    train_results["train_losses"],
    train_results["val_losses"],
    train_results["train_accuracies"],
    train_results["val_accuracies"]
)

# ---------------------- Test Evaluation ---------------------- #
# Best Model
best_model = models.build_multimodal_model(num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim).to(DEVICE)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

best_test_results = utils.evaluate(best_model, test_loader, DEVICE, criterion)
utils.save_test_metrics(
    metrics_file_path = METRICS_BEST_MODEL,
    balanced_accuracy = best_test_results["balanced_accuracy"],
    acc = best_test_results["acc"],
    precision = best_test_results["precision"],
    recall = best_test_results["recall"],
    f1 = best_test_results["f1"],
    roc_auc = best_test_results["roc_auc"],
    matthews = best_test_results["matthews"],
    specificity = best_test_results["specificity"],
    tnr=best_test_results["tnr"]
)
best_prob_scores = np.array(best_test_results["prob_scores"])
utils.save_predictions(
    predictions_file_path=PREDI_BEST_MODEL,
    y_true=best_test_results["targets"],
    y_pred=best_test_results["predictions"],
    prob_scores=best_prob_scores
)
class_names = ['Cesarean birth','Vaginal birth']  # label 1, label 0
utils.plot_confusionMatrix(CM_BEST_MODEL, best_test_results["confusion_matrix"], class_names, "Confusion Matrix: Cesarean vs Vaginal Birth")
utils.plot_RocCurve(ROC_BEST_MODEL,targets=best_test_results["targets"],prob_scores=best_test_results["prob_scores"])

# Final Model
final_model = models.build_multimodal_model(num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim).to(DEVICE)
final_model.load_state_dict(torch.load(FINAL_MODEL_PATH))

final_test_results = utils.evaluate(final_model, test_loader, DEVICE)
utils.save_test_metrics(
    metrics_file_path = METRICS_FINAL_MODEL,
    balanced_accuracy = final_test_results["balanced_accuracy"],
    acc = final_test_results["acc"],
    precision = final_test_results["precision"],
    recall = final_test_results["recall"],
    f1 = final_test_results["f1"],
    roc_auc = final_test_results["roc_auc"],
    matthews = final_test_results["matthews"],
    specificity = final_test_results["specificity"],
    tnr=final_test_results["tnr"]
)
final_prob_scores = np.array(final_test_results["prob_scores"])
utils.save_predictions(
    predictions_file_path=PREDI_FINAL_MODEL,
    y_true=final_test_results["targets"],
    y_pred=final_test_results["predictions"],
    prob_scores=final_prob_scores
)
class_names = ['Cesarean birth','Vaginal birth']  # label 1, label 0
utils.plot_confusionMatrix(CM_FINAL_MODEL, final_test_results["confusion_matrix"], class_names, "Confusion Matrix: Cesarean vs Vaginal Birth")
utils.plot_RocCurve(ROC_FINAL_MODEL,targets=final_test_results["targets"],prob_scores=final_test_results["prob_scores"])