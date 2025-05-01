import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TF_binder_model import TFBindingTransformer, TFBindingDataset
from  train_model import train_model
from evaluate_model import evaluate_model, plot_loss_curve
import json

# Define amino acid encoding dictionary
aa_to_index = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1-based index, 0 for padding

def encode_sequence(sequence, aa_to_index, max_length=50):
    """Convert sequence to token indices for transformer embedding"""
    if not isinstance(sequence, str):
        return [0] * max_length  # Return all padding if sequence is missing
    sequence = sequence[:max_length]  # Truncate long sequences
    return [aa_to_index.get(aa, 0) for aa in sequence] + [0] * (max_length - len(sequence))

def setup_logging(output_dir):
    """Configure logging to file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"=== HAT Score Model Training Log ===\n")
        f.write(f"Started training at: {pd.Timestamp.now()}\n\n")
    return log_file


def load_and_preprocess_data(data_path, domain):
    """Load, clean, and transform protein interaction data."""
    print(f"Loading data from {data_path}")
    df = pd.read_excel(data_path)
    # Filter the dataframe for a single domain
    df = df[df['Domain'] == domain]

    # Ensure 'Sequence' column is a string
    df['Sequence'] = df['Sequence'].astype(str)
    df['domain_sequence'] = df['domain_sequence'].astype(str)

    # Define feature groups
    af_features = [ 'iptm', 'average pae score' , 'mpDockQ/pDockQ']
    structure_features = [ 'int_solv_en' , 'Charged' , 'Hydrophobic' , 'contact_pairs', 'int_area']
    all_features = af_features + structure_features
    
    # Print the initial shape of the dataframe
    print(f"Initial data shape: {df.shape}")

    # Replace 'None' and other non-numeric strings with NaN
    df[all_features] = df[all_features].replace('None', np.nan)

    # Convert columns to numeric, coercing any errors to NaN
    df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values in any of the specified columns
    df = df.dropna(subset=all_features)

    # Print the final shape of the dataframe
    print(f"Final data shape after cleaning: {df.shape}")

    # Encode sequences
    df['encoded_sequence'] = df['Sequence'].apply(lambda seq: encode_sequence(seq, aa_to_index))

    df['encode_domain'] =df['domain_sequence'].apply(lambda seq: encode_sequence(seq, aa_to_index))

    # Normalize Kd values
    df["Transformed Kd (nM)"] = 1 - (df['Kd (nM)'] - df['Kd (nM)'].min()) / (df['Kd (nM)'].max() - df['Kd (nM)'].min())
    print(df['Kd (nM)'].max())
    print(df['Kd (nM)'].min())
    return df, af_features, structure_features


def prepare_features(df, structure_features, af_features, save_scalers=True):
    """Prepare and scale features for model training, with domain-specific normalization."""
    seq_features = np.array(df['encoded_sequence'].tolist())
    struct_features = df[structure_features].values  # Correctly use structure_features
    alphafold_features = df[af_features].values  # Correctly use af_features
    domain_sequences = np.array(df['encode_domain'].tolist())
    
    # Initialize arrays to store the scaled features
    struct_scaled = np.zeros_like(struct_features)
    alphafold_scaled = np.zeros_like(alphafold_features)

    struct_scaler = StandardScaler()
    alphafold_scaler = StandardScaler()

    struct_scaled = struct_scaler.fit_transform(struct_features)
    alphafold_scaled = alphafold_scaler.fit_transform(alphafold_features)
    
    if save_scalers:
        domain_scalers = {'structure': struct_scaler, 'alphafold': alphafold_scaler}
    
    return seq_features, struct_scaled, alphafold_scaled, df['Transformed Kd (nM)'].values, domain_sequences, domain_scalers

def main():
    domain = input("What domain you want to look at ").upper()
    output_dir = f'/Users/castroverdeac/Desktop/hat_score/making_model_individual_domain/{domain}'
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Path to the data file
    data_path = '/Users/castroverdeac/Desktop/hat_score/data/data_tfs_ep300_cbp_clean.xlsx'


    log_file = setup_logging(output_dir)
    df, af_features, structure_features = load_and_preprocess_data(data_path, domain)

    
    print(f"The number of items before spliting to train, test, and validate data: {df.shape}" )

    df.to_csv(os.path.join(output_dir, f'{domain}_processed_data.csv'), index=False)
    
    print("Preparing features...")
    # Correct parameter order
    seq_features, struct_scaled, alphafold_scaled, y, domain_sequences, domain_scalers = prepare_features(
        df, structure_features, af_features, save_scalers=True
    )
    
    # Save the scalers
    joblib.dump(domain_scalers, os.path.join(output_dir, f'{domain}_domain_scaler.joblib'))

    X = pd.DataFrame({
        'seq_features': list(seq_features),
        'struct_features': list(struct_scaled),
        'alphafold_features': list(alphafold_scaled),
        'domain_sequences': list(domain_sequences)
    })

    # Train
    model, val_df, test_dataset, history, x_index, traing_predictions = train_model(X=X, y=y, epochs=100)
    # Locate the indexes in the df from x_index
    located_indexes = df.iloc[x_index].index.tolist()
    new_df = df.loc[located_indexes]

    with open(f'{output_dir}/history_{domain}.txt', 'w') as f:
        json.dump(history, f, indent=4)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Then call evaluate_model with the correct order of arguments
    all_predictions, overall_metrics= evaluate_model(test_dataloader, model, lambda_l2=0.01)
    
    torch.save(model.state_dict(), f'{output_dir}/model_{domain}.pth')

    # Ensure new_df and all_predictions have matching indices
    new_df = new_df.reset_index(drop=True)
    all_predictions = pd.DataFrame(all_predictions, columns=['predictions']).reset_index(drop=True)

    # Combine new_df with all_predictions
    combined_df = pd.concat([new_df, all_predictions], axis=1)

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(os.path.join(output_dir, f'combined_predictions_{domain}.csv'), index=False)

    # Grabbing Validation 
    loc_val = df.iloc[val_df['index']].index.tolist()
    val_new_df = df.loc[loc_val]
    val_new_df = val_new_df.reset_index(drop=True)
    combine_val = pd.concat([val_new_df, val_df], axis=1)
    combine_val.to_csv(os.path.join(output_dir, 'validatiaon_df.csv'), index=False)

    # Save the matching training data
    loc_val = df.iloc[traing_predictions['index']].index.tolist()
    train_new_df = df.loc[loc_val]
    train_new_df = train_new_df.reset_index(drop=True)
    combine_train = pd.concat([train_new_df, traing_predictions], axis=1)
    combine_train.to_csv(os.path.join(output_dir, 'training_df.csv'), index=False)

    overall_metrics.to_csv(os.path.join(output_dir, f'metrics_{domain}.csv'), index=False)

    plot_loss_curve(history)

    print(f"Training complete. Models saved to {output_dir}")

if __name__ == "__main__":
    main()
