import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class TFBindingDataset(Dataset):
    """
    This is a Dataset for handling the data related to TFs and EP300 domain binding

    In this dataset, it is designed to store andprovide access to various of features
    labels associated with TF binding which includees sequence features, inferface structural
    features, AF predictions, and binding  affinity (output).

    This is compatible with PyTorch DataLoader for efficient batching and shuffling to train and 
    evaluate the machine learning model.
    """
    def __init__(self, seq_features, domain_seq_features, struct_features, alphafold_features, domain_indices, kd_values, difficulty=None):
        self.seq_features = torch.tensor(np.array(seq_features), dtype=torch.float32) #transcriptional factor sequence features
        self.domain_seq_features = torch.tensor(np.array(domain_seq_features), dtype=torch.float32) #domain sequence features
        self.struct_features = torch.tensor(np.array(struct_features), dtype=torch.float32) #interface structure features
        self.alphafold_features = torch.tensor(np.array(alphafold_features), dtype=torch.float32) #alphafold features
        self.domain_indices = torch.tensor(np.array(domain_indices), dtype=torch.long) #mapping sequencing to their respecitve domain
        self.kd_values = torch.tensor(np.array(kd_values), dtype=torch.float32) #outcome -> transformed binding affinity

    def __len__(self):
        return len(self.kd_values) # returns the number of samples in the dataset

    def __getitem__(self, idx):
        return (self.seq_features[idx], #retrieves the features and labels for the sample for the specific sample
                self.struct_features[idx],
                self.alphafold_features[idx], 
                self.domain_indices[idx], 
                self.kd_values[idx], 
                self.domain_seq_features[idx]) 
    

class DomainSpecificModel(nn.Module):
    """This is a shared trunk with a mini head that is unique for each domain
    Creates a new layer for each domain that has domain-specific bias parameters.
    
    This is a shared encoder that learns features that are common to every domain"""
    def __init__(self, input_size, num_domains):
        super(DomainSpecificModel, self).__init__()
        self.input_size = input_size
        self.num_domains = num_domains
        
        # Common shared layers
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        
        # Domain-specific head (domain bias branch)
        self.domain_heads = nn.ModuleList([
        nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout for domain components
            nn.Linear(128, 1, bias=False)
        ) for _ in range(num_domains)
    ])
    
    def forward(self, x, domain):
        x = self.relu(self.fc1(x))
        
        # Apply domain-specific head
        domain_output = self.domain_heads[domain](x)  # Select the correct head for the current domain
        
        return domain_output


class ResidualBlock(nn.Module):
    """ Implements a residual block with LayerNorm and dropout"""
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x):
        residual = x
        x = self.norm1(x) #normlaizes input across features which helps with gradient flow
        x = self.linear1(x) #increases dimensionality -> learns complex interactions
        x = F.gelu(x) # activate
        x = self.dropout(x) #contract
        x = self.linear2(x) #brings back to original dimension
        x = self.dropout(x)
        x = x + residual #preserves information  -> train deeper networks
        return x


class CrossAttention(nn.Module):
    """allows for the model to understand what pieces of the input are the most important for generating the output.

    The model asks the question what is the most important (query), finds an answer to the question (key), and then focuses on 
    the actual info (value)"""
    def __init__(self, query_dim, key_dim, embed_dim, num_heads=4, dropout=0.2):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim) #question 
        self.key_proj = nn.Linear(key_dim, embed_dim) #answer
        self.value_proj = nn.Linear(key_dim, embed_dim) #value
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        # First apply projections
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        
        # Then apply normalization
        q = self.norm(q).unsqueeze(1)  # Add a sequence dimension
        k = self.norm(k).unsqueeze(1)
        v = self.norm(v).unsqueeze(1)
        
        attn_output, _ = self.attention(q, k, v)
        return attn_output.squeeze(1)

class EnhancedFeatureProcessor(nn.Module):
    """"This process the input features"""
    def __init__(self, input_dim, output_dim, num_layers=2, dropout=0.1):
        super(EnhancedFeatureProcessor, self).__init__()
        self.input_layer = nn.Linear(input_dim, output_dim) #fully connected layer
        self.norm = nn.LayerNorm(output_dim) #stabalizing training
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(output_dim, dropout) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        input_x = x  # Save the input
        x = self.input_layer(x)
        x = self.norm(x)
        x = F.gelu(x) #experimental data highlights linearity if looking at all four domains; however, does not show within the same domain 
        
        for block in self.residual_blocks:
            x = block(x) #perserves info from earlier layers
        
        # If dimensions match, add a skip connection from input to output
        if input_x.shape[-1] != x.shape[-1]:
            input_x = self.input_layer(input_x)
        
        x = x + input_x
        
        return x
    
def add_noise(x, std=0.1):
    """Applies Gaussian noise to a tensor"""
    return x + torch.randn_like(x) * std

def random_mask(x, mask_prob=0.1):
    """
    Randomly masks a percentage of the input.
    Supports both [batch, seq_len, dim] and [batch, dim] shaped inputs.
    """
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_prob
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        return x * (~mask)
    elif x.dim() == 2:
        batch_size, dim = x.shape
        mask = torch.rand(batch_size, device=x.device) < mask_prob
        mask = mask.unsqueeze(-1)  # [batch, 1]
        return x * (~mask)
    else:
        raise ValueError(f"Unsupported input shape for masking: {x.shape}")


class TFBindingTransformer(nn.Module):
    """This is the main model class for the TFBindingTransformer"""
    def __init__(self, seq_feature_dim, domain_seq_dim, struct_dim, alphafold_dim, 
                 num_domains, embed_dim, hidden_dim, num_heads, 
                 num_layers, dropout=0.3):
        super(TFBindingTransformer, self).__init__()

        # Enhanced feature processors with residual connections
        self.seq_processor = EnhancedFeatureProcessor(seq_feature_dim, embed_dim)
        self.domain_seq_processor = EnhancedFeatureProcessor(domain_seq_dim, embed_dim)
        self.struct_processor = EnhancedFeatureProcessor(struct_dim, hidden_dim, num_layers=3)
        self.alphafold_processor = EnhancedFeatureProcessor(alphafold_dim, hidden_dim, num_layers=3)

        # Transformer Encoder with more capacity and layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,  # Increased capacity
            dropout=dropout,
            batch_first=True,
            activation="gelu"  # Better activation function
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Domain Embedding with more dimensions
        self.domain_embedding = nn.Embedding(num_domains, hidden_dim)
        self.domain_norm = nn.LayerNorm(hidden_dim)
        
        #Domain Bias
        self.domain_heads = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False)
                                   for _ in range(num_domains)])

        # Cross-attention mechanisms for feature interaction
        self.seq_alphafold_attention = CrossAttention(embed_dim, hidden_dim, hidden_dim)
        self.seq_struct_attention = CrossAttention(embed_dim, hidden_dim, hidden_dim)
        self.domain_seq_alphafold_attention = CrossAttention(embed_dim, hidden_dim, hidden_dim)

        # Layer normalization for feature combination
        self.combined_norm = nn.LayerNorm(embed_dim * 2 + hidden_dim * 3)
        
        # Output layers with increased capacity
        combined_dim = (embed_dim * 2) + (hidden_dim * 3)
        self.fc_hidden = nn.Linear(combined_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Linear(embed_dim * 2 + hidden_dim * 3, embed_dim * 2 + hidden_dim * 3) #takes combined features and outputs scalers for importance

    def forward(self, seq_input, struct_input, alphafold_input, domain_input, domain_seq_input):
        if self.training and torch.rand(1).item() < 0.8:  # Add noise during training with 80% probability
            # Add Gaussian noise to structured inputs
            struct_input = add_noise(struct_input, std=0.2)  # Increase noise standard deviation
            alphafold_input = add_noise(alphafold_input, std=0.2) 
            
            # Randomly mask sequence inputs
            seq_input = random_mask(seq_input, mask_prob=0.2)  # Increase masking probability
            domain_seq_input = random_mask(domain_seq_input, mask_prob=0.2)
            
            # Add random scaling noise
            scaling_factor = torch.randn_like(seq_input) * 0.1 + 1.0  # Scale inputs by a random factor
            seq_input = seq_input * scaling_factor
            domain_seq_input = domain_seq_input * scaling_factor
            
            # Add random dropout noise
            dropout_mask = torch.rand_like(seq_input) > 0.9  # 10% dropout
            seq_input = seq_input * (~dropout_mask)
            domain_seq_input = domain_seq_input * (~dropout_mask)

        # Process sequence features
        seq_features = self.seq_processor(seq_input)
        
        # Transformer encoding with residual connection
        seq_transformer_in = seq_features.unsqueeze(1)  # Add sequence dimension
        seq_transformer_out = self.transformer_encoder(seq_transformer_in).squeeze(1)
        seq_out = seq_transformer_out + seq_features  # Residual connection
        
        # Process domain sequence features
        domain_seq_features = self.domain_seq_processor(domain_seq_input)
        domain_seq_transformer_in = domain_seq_features.unsqueeze(1)
        domain_seq_transformer_out = self.transformer_encoder(domain_seq_transformer_in).squeeze(1)
        domain_seq_out = domain_seq_transformer_out + domain_seq_features  # Residual connection
        
        # Process structural and AlphaFold features with enhanced processors
        struct_out = self.struct_processor(struct_input)
        alphafold_out = self.alphafold_processor(alphafold_input)
        seq_struct_attn = self.seq_struct_attention(seq_out, struct_out)
        domain_alphafold_attn = self.domain_seq_alphafold_attention(domain_seq_out, alphafold_out)

        # Get domain embeddings
        domain_out = self.domain_embedding(domain_input)
        
        
        # Apply cross-attention between sequence and structural features
        seq_alphafold_attn = self.seq_alphafold_attention(seq_out, alphafold_out)
        
        # Combine all features with attention-weighted features
        struct_out = struct_out + seq_struct_attn  # Add attention influence
        alphafold_out = alphafold_out + seq_alphafold_attn + domain_alphafold_attn
        
        # Combine all features
        combined = torch.cat([
            seq_out,
            struct_out, 
            alphafold_out, 
            domain_out, 
            domain_seq_out
        ], dim=1)
        
        # Apply normalization to combined features
        gate = torch.sigmoid(self.fusion_gate(combined))
        combined = combined * gate  # Gated feature fusion
        combined = self.combined_norm(combined)
        

        # Process through final layers with residual connection
        hidden = F.gelu(self.fc_hidden(combined))
        hidden = self.output_norm(hidden)
        hidden = self.dropout(hidden)
        
        # Final output
        logit_base = self.fc_out(hidden).squeeze(1)  # [batch]
        # Ensure consistent shape for domain head outputs
        bias = torch.zeros_like(logit_base)
        for i, d in enumerate(domain_input):
            # Make sure this returns a scalar for each sample
            domain_output = self.domain_heads[d](hidden[i].unsqueeze(0)).view(-1)
            bias[i] = domain_output

        # Apply sigmoid to the sum
        output = torch.sigmoid(logit_base + bias)

        return output

def predict_scores(model, new_data, batch_size=32):
    """
    Predict Kd scores for new data using the trained model.
    
    :param model: The trained model
    :param new_data: A dictionary containing the new data features.
                     Must include keys: 'seq_features', 'domain_seq_features', 'struct_features', 
                     'alphafold_features', 'domain_indices'
    :param batch_size: Batch size for prediction
    :return: Predicted Kd scores
    """
    
    # Prepare the new data as tensors
    seq_features = torch.tensor(np.array(new_data['seq_features']), dtype=torch.float32)
    domain_seq_features = torch.tensor(np.array(new_data['domain_seq_features']), dtype=torch.float32)
    struct_features = torch.tensor(np.array(new_data['struct_features']), dtype=torch.float32)
    alphafold_features = torch.tensor(np.array(new_data['alphafold_features']), dtype=torch.float32)
    domain_indices = torch.tensor(np.array(new_data['domain_indices']), dtype=torch.long)
    
    # Create placeholder kd_values (not used for prediction)
    dummy_kd = torch.zeros(len(seq_features), dtype=torch.float32)
    
    # Create a DataLoader for the new data
    dataset = TFBindingDataset(
        seq_features, 
        domain_seq_features,
        struct_features, 
        alphafold_features, 
        domain_indices, 
        dummy_kd
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()  # Set the model to evaluation mode
    
    predictions = []
    with torch.no_grad():  # Disable gradient calculation
        for seq_input, struct_input, alphafold_input, domain_input, _, domain_seq_input in dataloader:
            output = model(seq_input, struct_input, alphafold_input, domain_input, domain_seq_input).squeeze()
            predictions.extend(output.tolist())
    
    return np.array(predictions)
