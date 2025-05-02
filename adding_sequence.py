import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import copy
import torch.nn as nn
import matplotlib as plt

def evaluate_model(model, val_dataset, test_dataset, batch_size=32, epoch_metrics=None):
    """
    Evaluate a model on validation and test datasets with enhanced metrics and visualization.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model to evaluate
    val_dataset : Dataset
        Validation dataset
    test_dataset : Dataset
        Test dataset
    batch_size : int
        Batch size for evaluation
    epoch_metrics : dict, optional
        Dictionary containing epoch-wise metrics from training
        Expected keys: 'train_losses', 'val_losses', 'val_overall_accs', 
                      'val_top25_accs', 'val_top50_accs'
    
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    # Create dataloaders for validation and testing datasets
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function
    criterion = nn.MSELoss()
    lambda_l2 = 0.01

    
    # Additional metrics
    metrics_functions = {
        'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        'binary_accuracy': lambda y_true, y_pred, threshold=0.1: np.mean(np.abs(y_true - y_pred) < threshold)
    }

    # Helper function to calculate comprehensive metrics
    def calculate_metrics(dataloader, dataset_name):
        model.eval()
        all_targets = []
        all_predictions = []
        total_loss = 0.0

        with torch.no_grad():
            for seq_input, struct_input, alphafold_input, domain_input, kd_target, domain_indices in dataloader:
                output = model(seq_input, struct_input, alphafold_input, domain_input, domain_indices).squeeze()
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss_fn = nn.MSELoss()
                loss = loss_fn(output, kd_target)
                loss += lambda_l2 * l2_norm
                total_loss += loss.item()

                all_targets.extend(kd_target.tolist())
                all_predictions.extend(output.tolist())

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Basic metrics
        results = {
            f"{dataset_name}_loss": total_loss / len(dataloader),
            f"{dataset_name}_mse": metrics_functions['mse'](all_targets, all_predictions),
            f"{dataset_name}_mae": metrics_functions['mae'](all_targets, all_predictions),
            f"{dataset_name}_r2": metrics_functions['r2'](all_targets, all_predictions),
        }
        
        # Calculate accuracy with threshold 0.1
        results[f"{dataset_name}_overall_acc"] = metrics_functions['binary_accuracy'](all_targets, all_predictions)
        
        # Calculate quartile-based metrics
        # For bottom 25%, middle 50%, and top 25%
        sorted_indices = np.argsort(all_targets)
        n = len(sorted_indices)
        
        # Bottom 25% (lowest values)
        bottom_25_indices = sorted_indices[:n//4]
        # Middle 50%
        middle_50_indices = sorted_indices[n//4:3*n//4]
        # Top 25% (highest values)
        top_25_indices = sorted_indices[3*n//4:]
        # Bottom 50%
        bottom_50_indices = sorted_indices[:n//2]
        # Top 50%
        top_50_indices = sorted_indices[n//2:]
        
        segments = {
            'bottom_25': bottom_25_indices,
            'middle_50': middle_50_indices,
            'top_25': top_25_indices,
            'bottom_50': bottom_50_indices,
            'top_50': top_50_indices
        }
        
        # Calculate metrics for each segment
        for segment_name, indices in segments.items():
            segment_targets = all_targets[indices]
            segment_predictions = all_predictions[indices]
            
            results[f"{dataset_name}_{segment_name}_mse"] = metrics_functions['mse'](segment_targets, segment_predictions)
            results[f"{dataset_name}_{segment_name}_mae"] = metrics_functions['mae'](segment_targets, segment_predictions)
            results[f"{dataset_name}_{segment_name}_acc"] = metrics_functions['binary_accuracy'](segment_targets, segment_predictions)
        
        return results, all_targets, all_predictions

    # Evaluate on validation and test datasets
    val_results, val_targets, val_predictions = calculate_metrics(val_dataloader, 'val')
    test_results, test_targets, test_predictions = calculate_metrics(test_dataloader, 'test')
    
    # Combine results
    all_results = {**val_results, **test_results}

    # Print detailed evaluation results
    print("\n" + "="*50)
    print(f"MODEL EVALUATION RESULTS")
    print("="*50)
    
    print("\nVALIDATION SET:")
    print(f"Loss (MSE): {val_results['val_loss']:.4f}")
    print(f"Mean Absolute Error: {val_results['val_mae']:.4f}")
    print(f"R² Score: {val_results['val_r2']:.4f}")
    print(f"Overall Accuracy (±0.1): {val_results['val_overall_acc']:.4f}")
    print(f"Top 25% Accuracy: {val_results['val_top_25_acc']:.4f}")
    print(f"Top 50% Accuracy: {val_results['val_top_50_acc']:.4f}")
    
    print("\nTEST SET:")
    print(f"Loss (MSE): {test_results['test_loss']:.4f}")
    print(f"Mean Absolute Error: {test_results['test_mae']:.4f}")
    print(f"R² Score: {test_results['test_r2']:.4f}")
    print(f"Overall Accuracy (±0.1): {test_results['test_overall_acc']:.4f}")
    print(f"Top 25% Accuracy: {test_results['test_top_25_acc']:.4f}")
    print(f"Top 50% Accuracy: {test_results['test_top_50_acc']:.4f}")
    print("="*50)

    # Set up a nice style for plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a more comprehensive visualization
    def plot_evaluation_results():
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. If we have epoch metrics, plot training history
        if epoch_metrics is not None and all(k in epoch_metrics for k in 
                                           ['train_losses', 'val_losses']):
            
            # Training history plot (losses)
            ax1 = plt.subplot2grid((2, 3), (0, 0))
            epochs = range(1, len(epoch_metrics['train_losses']) + 1)
            ax1.plot(epochs, epoch_metrics['train_losses'], 'b-', label='Training Loss')
            ax1.plot(epochs, epoch_metrics['val_losses'], 'r-', label='Validation Loss')
            ax1.set_title('Training History')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Training history plot (accuracy)
            if all(k in epoch_metrics for k in ['val_overall_accs', 'val_top25_accs', 'val_top50_accs']):
                ax2 = plt.subplot2grid((2, 3), (0, 1))
                ax2.plot(epochs, epoch_metrics['val_overall_accs'], 'g-', label='Overall Acc')
                ax2.plot(epochs, epoch_metrics['val_top25_accs'], 'm-', label='Top 25% Acc')
                ax2.plot(epochs, epoch_metrics['val_top50_accs'], 'c-', label='Top 50% Acc')
                ax2.set_title('Validation Accuracy During Training')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
        
        # 2. Scatter plot of predictions vs actual values
        ax3 = plt.subplot2grid((2, 3), (0, 2)) if epoch_metrics else plt.subplot2grid((2, 3), (0, 0), colspan=2)
        
        # Create a gradient of colors based on density
        from scipy.stats import gaussian_kde
        xy = np.vstack([val_targets, val_predictions])
        z = gaussian_kde(xy)(xy)
        
        # Sort points by density for better visualization
        idx = z.argsort()
        val_targets_sorted, val_predictions_sorted, z_sorted = val_targets[idx], val_predictions[idx], z[idx]
        
        scatter = ax3.scatter(val_targets_sorted, val_predictions_sorted, c=z_sorted, s=30, alpha=0.6, cmap='plasma')
        
        # Add perfect prediction line
        min_val = min(val_targets.min(), val_predictions.min())
        max_val = max(val_targets.max(), val_predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax3.set_title('Validation: Predicted vs Actual')
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        fig.colorbar(scatter, ax=ax3, label='Density')
        
        # 3. Error distribution histogram
        ax4 = plt.subplot2grid((2, 3), (1, 0))
        errors = val_predictions - val_targets
        ax4.hist(errors, bins=30, alpha=0.7, color='#3572EF', edgecolor='grey')
        ax4.axvline(x=0, color='r', linestyle='--')
        ax4.set_title('Validation: Error Distribution')
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        
        # 4. Segmented performance comparison (bar chart)
        ax5 = plt.subplot2grid((2, 3), (1, 1))
        segments = ['overall', 'bottom_25', 'middle_50', 'top_25']
        segment_names = ['Overall', 'Bottom 25%', 'Middle 50%', 'Top 25%']
        
        val_acc_values = [val_results[f'val_{seg}_acc'] for seg in segments]
        test_acc_values = [test_results[f'test_{seg}_acc'] for seg in segments]
        
        x = np.arange(len(segments))
        width = 0.35
        
        ax5.bar(x - width/2, val_acc_values, width, label='Validation', color='#87A2FF', edgecolor='grey')
        ax5.bar(x + width/2, test_acc_values, width, label='Test', color='#E78F81', edgecolor='grey')
        
        ax5.set_title('Accuracy by Data Segment')
        ax5.set_xlabel('Data Segment')
        ax5.set_ylabel('Accuracy (±0.1)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(segment_names)
        ax5.legend()
        
        # 5. Feature importance or model summary
        ax6 = plt.subplot2grid((2, 3), (1, 2))
        metrics = ['mse', 'mae', 'r2']
        metric_names = ['MSE', 'MAE', 'R²']
        
        val_metric_values = [val_results[f'val_{metric}'] for metric in metrics]
        test_metric_values = [test_results[f'test_{metric}'] for metric in metrics]
        
        x = np.arange(len(metrics))
        
        ax6.bar(x - width/2, val_metric_values, width, label='Validation', color='#87A2FF', edgecolor='grey')
        ax6.bar(x + width/2, test_metric_values, width, label='Test', color='#E78F81', edgecolor='grey')
        
        ax6.set_title('Performance Metrics')
        ax6.set_xlabel('Metric')
        ax6.set_ylabel('Value')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metric_names)
        ax6.legend()
        
        # Adjust layout and save/show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        plt.show()
    
    # Plot the evaluation results
    plot_evaluation_results()
    
    return all_results


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
        plt.show()
    else:
        print("Epoch metrics do not contain 'train_losses' or 'val_losses'.")