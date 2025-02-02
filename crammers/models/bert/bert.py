import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """Vanilla Multiheaded Attention Layer

    Args:
        heads (int): Number of heads
        d_model (int): Dimension of the model
        dropout (float): Dropout rate. Default: 0.1

    References:
        https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
    """

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Forward pass of the MultiHeadedAttention layer

        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            mask (torch.Tensor): Mask tensor

        Returns:
            torch.Tensor: Output tensor
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(
            query.size(-1)
        )
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, value)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )

        return self.output_linear(context)


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
