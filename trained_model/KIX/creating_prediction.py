import pandas as pd
import numpy as np
import joblib
import os 
import torch
from TF_binder_model import CrossAttention, ResidualBlock, EnhancedFeatureProcessor, TFBindingDataset, TFBindingTransformer, predict_scores

aa_to_index = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1-based index, 0 for padding
#--------------------- Definitions ---------------------#
def encode_sequence(sequence, aa_to_index, max_length=50):
    """Convert sequence to token indices for transformer embedding"""
    if not isinstance(sequence, str):
        return [0] * max_length  # Return all padding if sequence is missing
    sequence = sequence[:max_length]  # Truncate long sequences
    return [aa_to_index.get(aa, 0) for aa in sequence] + [0] * (max_length - len(sequence))

def load_and_preprocess_data(df):
    df['Sequence'] = df['Sequence'].astype(str)
    df['domain_sequence'] = df['domain_sequence'].astype(str)

    # Define feature groups
    af_features = [ 'iptm', 'average pae score' , 'mpDockQ/pDockQ']
    structure_features =  [ 'int_solv_en' , 'Charged' , 'Hydrophobic' , 'contact_pairs', 'int_area']
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
    df['encode_domain'] = df['domain_sequence'].apply(lambda seq: encode_sequence(seq, aa_to_index))

    return df, af_features, structure_features


def prepare_features(df, structure_features, af_features, existing_scalers):
    """Prepare and scale features for model prediction using existing scalers."""
    seq_features = np.array(df['encoded_sequence'].tolist())
    struct_features = df[structure_features].values
    alphafold_features = df[af_features].values
    domain_sequences = np.array(df['encode_domain'].tolist())
    
    struct_scaler = existing_scalers['structure']
    alphafold_scaler = existing_scalers['alphafold']

    # Debugging
    print(f"Structure scaler trained with {struct_scaler.n_features_in_} features")
    print(f"AlphaFold scaler trained with {alphafold_scaler.n_features_in_} features")

    # Transform features using the correct scalers
    struct_scaled = struct_scaler.transform(struct_features)
    alphafold_scaled = alphafold_scaler.transform(alphafold_features)

    return seq_features, struct_scaled, alphafold_scaled, domain_sequences

#--------------------- Loading Data ---------------------#
file_path = input("Please enter the path to your CSV or Excel file: ")

if file_path.endswith('.csv'):
    data = pd.read_csv(file_path)
elif file_path.endswith(('.xls', '.xlsx')):
    data = pd.read_excel(file_path)
else:
    raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

print("Data successfully loaded.")

#--------------------- Grabbing scalers ---------------------#
scaler_path = os.path.join(os.path.dirname(__file__), 'KIX_domain_scaler.joblib')

if os.path.exists(scaler_path):
    existing_domain_scaler = joblib.load(scaler_path)
    print("Scalers successfully loaded.")
else:
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
print(existing_domain_scaler)

#--------------------- Preparing Data ---------------------#
df, af_features, structure_features = load_and_preprocess_data(data)
seq_features, struct_scaled, alphafold_scaled, domain_sequences = prepare_features(
    df, structure_features, af_features, existing_domain_scaler
)

#--------------------- Defining Model ---------------------#
seq_feature_dim = seq_features.shape[1]
domain_seq_dim = domain_sequences.shape[1]
struct_dim = struct_scaled.shape[1]
alphafold_dim = alphafold_scaled.shape[1]

embed_dim = 64
hidden_dim = 128
num_heads = 4
num_layers = 2

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model_KIX.pth')
if os.path.exists(model_path):
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
    model.load_state_dict(torch.load(model_path))
    print("Model successfully loaded.")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

#--------------------- Making a Prediction ---------------------#
# Prepare the data in the format expected by predict_scores
new_data = {
    'seq_features': seq_features,
    'domain_seq_features': domain_sequences,
    'struct_features': struct_scaled,
    'alphafold_features': alphafold_scaled}

# Make predictions
predicted_scores = predict_scores(model, new_data)

#--------------------- Output ---------------------#
file_name = input("What are you predicting? ")
df['Predicted_Scores'] = predicted_scores

file_path_1 = os.path.join(os.path.dirname(__file__), 'predictions')

# Ensure the directory exists
os.makedirs(file_path_1, exist_ok=True)

# Save the updated dataframe to a new CSV file
output_path = os.path.join((file_path_1), f'{file_name}_predictions_output.csv')
df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
