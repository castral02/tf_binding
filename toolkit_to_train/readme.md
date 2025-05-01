# Training your own model
In this repository, we give you the tools to run and create your own model. 

## Model Architecture Overview
*Enhnaced Feature Processing*
The inputs of AlphaFold Metrics, transcription factor sequence, domain sequence, domain category, interface structure metrics are transformed to a high dimensional representation in ```Enhanced Feature Processing```.

Feautres are first passed through a fully connected linear that projects them from an input dimension to an output dimension. After this, features are normalized using a ```LayerNorm```. These normalized features are then passed through a GELU (Gaussian Error Linear Unit) activation, which softly weights input values. The activated features are processed by a layer of residual bloacks for the model to learn deeper transformations and preserving the original information. 

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
