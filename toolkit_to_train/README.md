# Training your own model
In this repository, we give you your own model, and an example on how to train your own model. 

If you want an explanation of how we trained our own model, please [click here](../trained_model)

## Model Architecture Overview

*Enhnaced Feature Processing*

The inputs of AlphaFold Metrics, transcription factor sequence, domain sequence, domain category, interface structure metrics are transformed to a high dimensional representation in ```Enhanced Feature Processing```.

1. **Linear Projection**: Features are first passed through a fully connected linear that projects them from the original dimension (```input_dim```) to a higher dimensional space (```output_dim```) output dimension.
2. **Layer Normalization**: After this, features are normalized using a ```LayerNorm```.
3. **GELU Activation**: These normalized features are then passed through a GELU (Gaussian Error Linear Unit) activation, which softly weights input values.
4. **Residual Blocks**: The activated features are processed by a layer of residual blocks for the model to learn deeper transformations and preserving the original information. 

```python 
class EnhancedFeatureProcessor(nn.Module):
    """"Processes the input features."""
    def __init__(self, input_dim, output_dim, num_layers=2, dropout=0.1):
        super(EnhancedFeatureProcessor, self).__init__()
        self.input_layer = nn.Linear(input_dim, output_dim)  # Project input_dim -> output_dim
        self.norm = nn.LayerNorm(output_dim) #helps with the stabilization and convergence of training
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(output_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.input_layer(x)  # Always project at the beginning
        x = self.norm(x)
        x = F.gelu(x) #helps with a smoother and more expressive activation

        for block in self.residual_blocks: #deepends the network 
            x = block(x)

        return x
```

*Cross Attention*

This module creates contextual relationship between different features by apply cross-attention helping the model learn about what parts of the input are most relevant to another. 

In this architecture, cross attention is used to model the folling interactions, 
1. Transcription factor sequences and interface structural features
2. Transcription factor sequence and AlphaFold Metrics
3. Domain sequences and AlphaFold Metrics

**How it works**:
This is a classic mechanism called [query-key-value (QKV)](https://poloclub.github.io/transformer-explainer/).

- Query (Q): the question; what we are focusing on?
- Key (K): the reference; what portions of the data is relevant?
- Value (v): the content; what information should be retrieved and passed forward?

```python
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
```

*Transformation Encoding*

A shared transformer encoder is used for both transcription factor and domain sequences, allowing the model to learn the contextual representations of these sequences. To further this learning, we added a residual connection to retain the original sequence information to stabalize learning. 

```python
#in the def __init__...
Transformer Encoder with more capacity and layers
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 2,  # Increased capacity
        dropout=dropout,
        batch_first=True,
        activation="gelu"  # Better activation function
        )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #this is shared between domain and tf to preserve raw sequence info 

#in the forward
    #Process transcription factor sequence features
    seq_transformer_in = seq_features.unsqueeze(1)  # Add sequence dimension
    seq_transformer_out = self.transformer_encoder(seq_transformer_in).squeeze(1)
    seq_out = seq_transformer_out + seq_features  # Residual connection
        
    # Process domain sequence features
    domain_seq_features = self.domain_seq_processor(domain_seq_input)
    domain_seq_transformer_in = domain_seq_features.unsqueeze(1)
    domain_seq_transformer_out = self.transformer_encoder(domain_seq_transformer_in).squeeze(1)
    domain_seq_out = domain_seq_transformer_out + domain_seq_features  # Residual connection
```

*Feature Fusion*

Before the final layers, all the processed features are fused toether into a single representation. This fusion enables the model to integrate multiple modalities-- such as sequence, structure, and AlphaFold-metrics-- into a unified view for downstream prediction. 

A fusion gate dynamically learn the importance of each modality essentially amplifying informative signal while supressing irrelevant noise. 

```python
#in the def __init__...
    self.fusion_gate = nn.Linear(combined_dim, combined_dim)

#in the forward
    # Combine all features
    combined = torch.cat([
        seq_out,
        struct_out,
        alphafold_out,
        domain_seq_out], dim=1)
    # Apply normalization to combined features
    gate = torch.sigmoid(self.fusion_gate(combined) #emphasize more important and informative inputs and down-weights less relevant ones based on the current context -> taking out noise
    combined = combined * gate  # Gated feature fusion
    combined = self.combined_norm(combined)
```

*Output Layer*

The final layers consist of a fully connected layer followed by a GELU activation and a dropout for regulization. 

A domain-specific bias is then added to capture the differences in binding behavior across the different domains. The output is passed through a sigmoid activation, ensuring the final score to be between 0-1. 
