import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(test_dataset, model, lambda_l2):
    """
    Evaluates model performance across different domains and data quartiles.
    
    Args:
        test_dataset: DataLoader or iterable containing evaluation data
        model: The model to evaluate
        lambda_l2: L2 regularization strength
    
    Returns:
        Tuple of predictions array and overall metrics dictionary
    """

    model.eval()
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    criterion = BalancedTopWeightedMSE(tau=0.75, alpha=3.0, gamma=3.0)
    
    with torch.no_grad():
        for seq_input, struct_input, alphafold_input, kd_target, domain_seq_input in test_dataset:
            output = model(seq_input, struct_input, alphafold_input, domain_seq_input).squeeze()
            
            # Convert to tensors/arrays for overall metrics
            all_predictions.append(output.cpu().numpy())
            all_targets.append(kd_target.cpu().numpy())

            # Calculate loss with L2 regularization
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = criterion(output, kd_target)
            loss += lambda_l2 * l2_norm
            total_loss += loss.item()

    # Flatten after all batches
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Calculate overall model metrics
    mse = np.mean((all_targets - all_predictions) ** 2)
    mae = np.mean(np.abs(all_targets - all_predictions))
    r2 = 1 - np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
    accuracy = np.mean(np.abs(all_targets - all_predictions) < 0.1)
    avg_loss = total_loss / len(test_dataset)
    
    overall_metrics = pd.DataFrame([{
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'avg_loss': avg_loss
    }])
    
    return all_predictions, overall_metrics


def plot_loss_curve(epoch_metrics):
    """
    Plot the training and validation loss curves over epochs.
        
    Parameters:
        -----------
    epoch_metrics : dict
    Dictionary containing epoch-wise metrics from training.
    Expected keys: 'train_losses', 'val_losses'
    """
    if 'train_losses' in epoch_metrics and 'val_losses' in epoch_metrics:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(epoch_metrics['train_losses']) + 1)
        plt.plot(epochs, epoch_metrics['train_losses'], label='Training Loss', color='#0D92F4', marker='o')
        plt.plot(epochs, epoch_metrics['val_losses'], label='Validation Loss', color='#C62E2E', marker='s')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/Users/castroverdeac/Desktop/hat_score/making_model/loss.jpg')
        plt.show()
    else:
        print("Epoch metrics do not contain 'train_losses' or 'val_losses'.")
