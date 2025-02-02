import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int, dropout_rate: float = 0.1):
        """Feed Forward Network that projects the contextualized representations to intermediate
        latent space and then back to the original space.

        Args:
            d_model (int): Dimension of the model
            intermediate_dim (int): Intermediate dimension
            dropout_rate (float): Dropout rate. Default: 0.1

        """
        super(FFN, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(d_model, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, d_model)

    # TODO: Add type hints
    def forward(self, batch):
        """Forward pass of the FFN layer"""

        batch = self.fc1(batch)
        batch = self.fc2(self.dropout(batch))
        return batch
