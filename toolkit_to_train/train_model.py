import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class BalancedTopWeightedMSE(torch.nn.Module):
    def __init__(self, tau, alpha, gamma, balance=0.7):
        """
        tau     : start of the top slice (0‑1). 0.75 → top‑25 %.
        alpha   : extra weight for the very top sample (alpha+1 is max).
        gamma   : base weight for the rest
        balance : balance between top domain accuracy and overall domain accuracy
        """
        super(BalancedTopWeightedMSE, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.balance = balance

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Get the top 'tau' percentage of predictions
        sorted_indices = torch.argsort(pred, descending=True)
        top_k = int(len(pred) * self.tau)
        
        # Assign higher weight to the top-k predictions
        weights = torch.zeros_like(target)
        weights[sorted_indices[:top_k]] = self.alpha  # Set top-k to alpha weight
        weights[sorted_indices[top_k:]] = self.gamma  # Set the rest to gamma weight
        
        # Compute the weighted mse
        weighted_mse = mse * weights
        
        # Top loss: mean of the top-k weighted losses
        top_loss = weighted_mse[weights == self.alpha].mean()
        
        # Overall loss: mean of all weighted losses
        overall_loss = weighted_mse.mean()

        # Return the combined loss with balance between top and overall
        return self.balance * top_loss + (1 - self.balance) * overall_loss

def train_model(X, y, epochs=100, patience=10):
    # Split the dataset into training, validation, and testing sets
    quartile = pd.qcut(y, q=4, labels=False, duplicates='drop')          
    strat_label = quartile.astype(str)

    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, strat_label,
        test_size=0.20,           
        random_state=42,
        stratify=strat_label      
    ) #splitting the dataset to be train and test/validate -> making sure the same amount of upper quartile and domains are the same

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,          
    random_state=42, stratify=strat_temp) # Split temp into validation and test, **stratifying on domain‑quartile labels**
    # so both sets keep the same distribution and we’re not extrapolating.

    # Extract features from X
    seq_features = X_train['seq_features'].tolist()
    domain_seq_features = X_train['domain_sequences'].tolist()
    struct_features = X_train['struct_features'].tolist()
    alphafold_features = X_train['alphafold_features'].tolist()

    # Create training dataset and dataloader
    train_dataset = TFBindingDataset(
        seq_features, 
        domain_seq_features,
        struct_features, 
        alphafold_features, 
        y_train
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Extract feature dimensions
    seq_feature_dim = train_dataset.seq_features.shape[1]
    domain_seq_dim = train_dataset.domain_seq_features.shape[1]
    struct_dim = train_dataset.struct_features.shape[1]
    alphafold_dim = train_dataset.alphafold_features.shape[1]

    # Initialize the model
    embed_dim = 64
    hidden_dim = 128
    num_heads = 4
    num_layers = 2

    model = TFBindingTransformer(
        seq_feature_dim=seq_feature_dim,
        domain_seq_dim=domain_seq_dim,
        struct_dim=struct_dim,
        alphafold_dim=alphafold_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Define loss function and optimizer
    criterion = BalancedTopWeightedMSE(tau=0.75, alpha=3.0, gamma=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Now we're maximizing top 25% accuracy 
        factor=0.7,
        patience=patience, #threshold if it does not learn something new
        threshold=0.01, #this is the improvement in learning 
        verbose=True #providing feedback
    ) #learning rate scheduler that reduces the learning rate when monitored to optimize the 
    #top 25%
    
    lambda_l2 = 0.01
    train_losses = []
    val_losses = []
    val_overall_accs = []
    val_top25_accs = []
    val_top50_accs = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0

    # Create validation dataset
    val_dataset = TFBindingDataset(
        X_val['seq_features'].tolist(),
        X_val['domain_sequences'].tolist(),
        X_val['struct_features'].tolist(),
        X_val['alphafold_features'].tolist(),
        y_val
    )
    
    # Create validation dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        for seq_input, struct_input, alphafold_input, kd_target, domain_seq_input in train_dataloader:
            optimizer.zero_grad()
            output = model(seq_input, struct_input, alphafold_input, domain_seq_input).squeeze()
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = criterion(output, kd_target)
            loss += lambda_l2 * l2_norm
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_targets.extend(kd_target.tolist())
            all_predictions.extend(output.tolist())

        # Calculate accuracy metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        overall_accuracy = np.mean(np.abs(all_targets - all_predictions) < 0.1)

        # Corrected: use negative sort for top values (highest values)
        sorted_indices = np.argsort(-all_targets)
        top_25_indices = sorted_indices[:len(sorted_indices) // 4]
        top_25_accuracy = np.mean(np.abs(all_targets[top_25_indices] - all_predictions[top_25_indices]) < 0.1)

        train_losses.append(total_loss / len(train_dataloader))

        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, "
                  f"Overall Accuracy: {overall_accuracy:.4f}, Top 25% Accuracy: {top_25_accuracy:.4f}")
        
        # Validation every epoch
        model.eval()
        val_all_targets = []
        val_all_predictions = []
        val_total_loss = 0.0

        with torch.no_grad():
            for seq_input, struct_input, alphafold_input, kd_target, domain_seq_input in val_dataloader:
                output = model(seq_input, struct_input, alphafold_input, domain_seq_input).squeeze()
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = criterion(output, kd_target)
                loss += lambda_l2 * l2_norm
                val_total_loss += loss.item()
                
                val_all_targets.extend(kd_target.tolist())
                val_all_predictions.extend(output.tolist())
        
        # Calculate validation metrics
        val_loss = val_total_loss / len(val_dataloader)
        val_losses.append(val_loss)

        val_all_targets = np.array(val_all_targets)
        val_all_predictions = np.array(val_all_predictions)
        
        # Overall accuracy
        val_overall_acc = np.mean(np.abs(val_all_targets - val_all_predictions) < 0.1)
        val_overall_accs.append(val_overall_acc)
        
        # For top percentiles, sort in descending order
        sorted_indices = np.argsort(-val_all_targets)  # Note the negative sign for descending order
        top_25_indices = sorted_indices[:len(sorted_indices) // 4]
        top_50_indices = sorted_indices[:len(sorted_indices) // 2]
        
        val_top25_acc = np.mean(np.abs(val_all_targets[top_25_indices] - val_all_predictions[top_25_indices]) < 0.1)
        val_top50_acc = np.mean(np.abs(val_all_targets[top_50_indices] - val_all_predictions[top_50_indices]) < 0.1)
        
        val_top25_accs.append(val_top25_acc)
        val_top50_accs.append(val_top50_acc)

        # Average top-25% accuracy across all domains
        val_top25_accs.append(val_top25_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
                  f"Val Overall Acc: {val_overall_acc:.4f}, Val Top 25% Acc: {val_top25_acc:.4f}")
        
        # Early stopping
        """if val_top25_acc < best_val_loss:
            best_val_loss = val_top25_acc
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break"""

        scheduler.step(val_top25_acc)
    
    # Use the best model for the final evaluation
    if best_model is not None:
        model = best_model
    
    # Create test dataset
    test_dataset = TFBindingDataset(
        X_test['seq_features'].tolist(),
        X_test['domain_sequences'].tolist(),
        X_test['struct_features'].tolist(),
        X_test['alphafold_features'].tolist(),
        y_test
    )

    # Return model and datasets
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_overall_accs': val_overall_accs,
        'val_top25_accs': val_top25_accs,
        'val_top50_accs': val_top50_accs}
    
    if epoch == epochs - 1:  # Only save predictions for the last epoch
        traing_predictions = pd.DataFrame({
            'true_values': all_targets,
            'predictions': all_predictions,
            'index': X_train.index
        })
        val_predictions_df = pd.DataFrame({
            'true_values': val_all_targets,
            'predictions': val_all_predictions,
            'index': X_val.index
        })
    
    return model, val_predictions_df, test_dataset, history, X_test.index, traing_predictions
