import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(train_path, val_path, test_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    return train_data, val_data, test_data

def shuffle_data(train_data, val_data):
    train_data = train_data.sample(frac=1, random_state=123).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=123).reset_index(drop=True)
    return train_data, val_data

def save_predictions(predictions, test_data, save_path):
    predictions_df = pd.DataFrame({
        'Class': test_data['Class'],        # True labels (ground truth)
        'Predictions': predictions,         # Predicted labels by the model
        'Processo': test_data['Processo']   # Process ID 
    })
    predictions_df.to_csv(save_path, index=False)
    print("Predictions saved to :", save_path)

def save_eval(scores, save_path):
    scores_df = pd.DataFrame(scores, index=[0])
    scores_df.to_csv(save_path, index=False)
    print("Evaluation results saved to: ", save_path) 
    
def plot_cm(cm, save_path):
    row_sums = np.sum(cm, axis=1, keepdims=True)
    cm_percent = cm / row_sums * 100
    labels = np.asarray([f'{value}\n({percent:.2f}%)' for value, percent in zip(cm.flatten(), cm_percent.flatten())]).reshape(2, 2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=['Cesarean birth','Vaginal birth'], yticklabels=['Cesarean birth','Vaginal birth'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print("Confusion matrix saved as a figure to: ", save_path)
    #plt.show()
    plt.close()
