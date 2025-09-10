import torch
import time
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, auc

def train_and_validate(multimodal_model, train_loader, val_loader, optimizer, criterion, device, best_model_path, final_model_path, epochs, patience):
    # Metric histories
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_epoch = 0
    best_val_loss = float('inf') # for Early Stopping with validation loss
    epochs_since_improvement = 0 # Early Stopping counter

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        multimodal_model.train()
        total_loss, correct, total = 0, 0, 0
        train_targets, train_preds = [], []

        for batch in train_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            labels = batch["label"]
            logits = multimodal_model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(acc)

        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Acc {acc:.4f}")

        # Validation
        multimodal_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_targets, val_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                labels = batch["label"]
                logits = multimodal_model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early Stopping check (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_since_improvement = 0  # Reset counter
            torch.save(multimodal_model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in val loss for {epochs_since_improvement} epoch(s).")

        # If patience exceeded, break
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (patience={patience}).")
            break

    # Save final model
    torch.save(multimodal_model.state_dict(), final_model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute confusion matrices for last epoch
    train_conf_matrix = confusion_matrix(train_targets, train_preds, labels=[1, 0])
    val_conf_matrix = confusion_matrix(val_targets, val_preds, labels=[1, 0])

    print(f"Best model saved at epoch: {best_epoch+1}")
    print(f"Training took {elapsed_time:.2f} seconds.")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_conf_matrix": train_conf_matrix,
        "val_conf_matrix": val_conf_matrix,
        "best_epoch": best_epoch,
        "elapsed_time": elapsed_time
    }

def train_only(multimodal_model, train_loader, optimizer, criterion, device, model_path, epochs):
    train_losses = []
    train_accuracies = []
    train_targets, train_preds = [], []

    start_time = time.time()

    for epoch in range(epochs):
        multimodal_model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_targets, epoch_preds = [], []

        for batch in train_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            labels = batch["label"]
            logits = multimodal_model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            epoch_preds.extend(preds.cpu().numpy())
            epoch_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Acc {acc:.4f}")

        # Store last epoch's predictions for confusion matrix
        train_targets = epoch_targets
        train_preds = epoch_preds

    # Save final model
    torch.save(multimodal_model.state_dict(), model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute confusion matrix for last epoch
    train_conf_matrix = confusion_matrix(train_targets, train_preds, labels=[1, 0])

    print(f"Training took {elapsed_time:.2f} seconds.")

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_conf_matrix": train_conf_matrix,
        "elapsed_time": elapsed_time
    }
        
#Evaluate model on Test dataset
def evaluate(multimodal_model, test_loader, device):
    multimodal_model.to(device)
    multimodal_model.eval()

    predictions = []
    targets = []
    prob_scores = []

    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            labels = batch["label"]
            outputs = multimodal_model(batch)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            prob_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())

    # Metrics
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=1)
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    matthews = matthews_corrcoef(targets, predictions)
    test_conf_matrix = confusion_matrix(targets, predictions, labels=[1, 0])
    roc_auc = roc_auc_score(targets, np.array(prob_scores)[:, 1])

    # TN, FP, FN, TP
    tp, fn, fp, tn = test_conf_matrix.ravel()
    tpr = tp / (tp + fp)
    tnr = tn / (tn + fn)
    b_acc = (tpr + tnr) / 2
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)

    # Print summary
    print(f"Balanced Accuracy: {balanced_accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Matthews Corrcoef: {matthews:.4f} | ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{test_conf_matrix}")

    return {
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews": matthews,
        "roc_auc": roc_auc,
        "confusion_matrix": test_conf_matrix,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "tpr": tpr, "tnr": tnr, "b_acc": b_acc, "acc": acc, "specificity": specificity,
        "predictions": predictions,
        "targets": targets,
        "prob_scores": prob_scores
    }

# Save the configuration into a config.json file
def save_config_file (config_file_path,optimizer, learning_rate, batch_size, epochs):
    config = {
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    with open(config_file_path, 'w') as config_file:
        json.dump(config, config_file,indent=1)
    print(f"Config saved to {config_file_path}")

# Save the training results into a history.json file
def save_train_metrics(history_file_path, train_losses, val_losses, train_accuracies, val_accuracies, best_epoch, elapsed_time):
    history = {
        "Train Losses": train_losses,
        "Validation Losses": val_losses,
        "Train Accuracies": train_accuracies,
        "Validation Accuracies": val_accuracies,
        "Best model saved after epoch": best_epoch,
        "Time": elapsed_time,
    }
    with open(history_file_path, 'w') as history_file:
        json.dump(history, history_file,indent=1)
    print(f"History saved to {history_file_path}")

# Save the continued training results into a history.json file
def save_continued_train_metrics(history_file_path, train_losses, train_accuracies, elapsed_time):
    history = {
        "Train Losses": train_losses,
        "Train Accuracies": train_accuracies,
        "Time": elapsed_time,
    }
    with open(history_file_path, 'w') as history_file:
        json.dump(history, history_file,indent=1)
    print(f"History saved to {history_file_path}")

# Save the model test results into a metrics.json file
def save_test_metrics(metrics_file_path, balanced_accuracy, acc, precision, recall, f1, roc_auc, matthews, specificity,tnr):
    metrics = {
        "Balanced Accuracy": balanced_accuracy,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "roc auc": roc_auc,
        "Matthews corrcoef": matthews,
        "Specificity": specificity,
        "NPV": tnr
    }
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file,indent=1)
    print(f"Metrics saved to {metrics_file_path}")

# Save the true class and predicted class into a pred_probs.csv file
def save_predictions(predictions_file_path, y_true, y_pred, prob_scores):
    prob_scores = np.array(prob_scores)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba_0": prob_scores[:, 0],
        "y_proba_1": prob_scores[:, 1]
    })
    df.to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")

#Plot train and validation loss and accuracy vs epochs
def plot_TrainVal_LossAcc(plots_file_path,train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(15, 7))

    #loss vs epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    #accuracy vs epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig(plots_file_path)
    print(f"Plots saved to {plots_file_path}")

#Plot train loss and accuracy vs epochs
def plot_Train_LossAcc(plots_file_path, train_losses, train_accuracies):
    plt.figure(figsize=(12, 5))

    # Loss vs epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()

    # Accuracy vs epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_file_path)
    print(f"Plots saved to {plots_file_path}")

# Plot confusion matrix
def plot_confusionMatrix(cm_file_path, conf_matrix, class_names, title_fig):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title_fig)
    plt.tight_layout()
    plt.savefig(cm_file_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {cm_file_path}")

# Plot Roc curve
def plot_RocCurve(roc_file_path, targets, prob_scores):
    fpr, tpr, thresholds = roc_curve(targets, np.array(prob_scores)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_file_path)
    plt.close()
    print(f"ROC curve saved to {roc_file_path}")