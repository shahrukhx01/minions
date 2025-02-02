from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import lightning as L

from crammers.optimizers.scheduled_optimizer import ScheduledOptim


@dataclass
class BertConfig:
    """Configuration class for BERT model parameters.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Dimension of the embeddings.
        seq_len (int): Maximum sequence length for the input.
        heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        feed_forward_hidden (int): Hidden layer size of the feedforward network.
        dropout (float): Dropout rate to prevent overfitting.
        pad_token_id (int): Token ID for padding.
        n_layers (int): Number of layers in the encoder.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        warmup_steps (int): Number of steps for the learning rate warmup.
    """
    def __init__(
        self,
        vocab_size=30522,
        embed_size=768,
        seq_len=512,
        heads=12,
        d_model=768,
        feed_forward_hidden=3072,
        dropout=0.1,
        pad_token_id=0,
        n_layers=12,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=4000,
    ):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.heads = heads
        self.d_model = d_model
        self.feed_forward_hidden = feed_forward_hidden
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps


class BERTEmbedding(torch.nn.Module):
    """BERT embedding layer that generates token, positional, and segment embeddings.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embed_size = config.embed_size
        self.token = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_token_id)
        self.segment = nn.Embedding(3, config.embed_size, padding_idx=config.pad_token_id)
        self.position = nn.Embedding(config.seq_len, config.embed_size, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, sequence, segment_label):
        """Generates embeddings by adding token, positional, and segment embeddings.

        Args:
            sequence (torch.Tensor): Tensor of input token indices.
            segment_label (torch.Tensor): Tensor of segment indices.

        Returns:
            torch.Tensor: Output embeddings after adding token, position, and segment embeddings.
        """
        batch = (
            self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        )
        return self.dropout(batch)


class MultiHeadedAttention(nn.Module):
    """Multi-head attention layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super(MultiHeadedAttention, self).__init__()
        assert config.d_model % config.heads == 0
        self.d_k = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = torch.nn.Dropout(config.dropout)

        self.query = torch.nn.Linear(config.d_model, config.d_model)
        self.key = torch.nn.Linear(config.d_model, config.d_model)
        self.value = torch.nn.Linear(config.d_model, config.d_model)
        self.output_linear = torch.nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Forward pass for multi-headed attention layer.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor for attention.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
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
    """Feed-forward network for the BERT encoder layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super(FFN, self).__init__()
        self.intermediate_dim = config.feed_forward_hidden
        self.dropout = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.d_model, self.intermediate_dim)
        self.fc2 = nn.Linear(self.intermediate_dim, config.d_model)

    def forward(self, batch):
        """Forward pass through the feed-forward network.

        Args:
            batch (torch.Tensor): Input tensor representing activations from previous layers.

        Returns:
            torch.Tensor: Output tensor after feed-forward operations.
        """
        batch = self.fc1(batch)
        batch = self.fc2(self.dropout(batch))
        return batch


class BertEncoderLayer(torch.nn.Module):
    """A single encoder layer in the BERT architecture.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super(BertEncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(config.d_model)
        self.self_multihead = MultiHeadedAttention(config)
        self.feed_forward = FFN(config)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single encoder layer.

        Args:
            embeddings (torch.Tensor): Input embeddings.
            mask (torch.Tensor): Mask tensor for attention.

        Returns:
            torch.Tensor: Output embeddings after applying multi-head attention and feed-forward network.
        """
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        return self.layernorm(feed_forward_out + interacted)


class Bert(nn.Module):
    """Vanilla BERT model implementation with multiple transformer layers.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.heads = config.heads
        self.feed_forward_hidden = config.d_model * 4
        self.embedding = BERTEmbedding(config)
        self.encoder_blocks = torch.nn.ModuleList([BertEncoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, batch, segment_labels):
        """Forward pass for the entire BERT model.

        Args:
            batch (torch.Tensor): Input token indices.
            segment_labels (torch.Tensor): Segment labels (e.g., sentence A/B).

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        mask = (batch > 0).unsqueeze(1).repeat(1, batch.size(1), 1).unsqueeze(1)
        batch = self.embedding(batch, segment_labels)

        for encoder in self.encoder_blocks:
            batch = encoder.forward(batch, mask)
        return batch


class NextSentencePrediction(torch.nn.Module):
    """2-class classification model: is_next, is_not_next

    Args:
        hidden (int): BERT model output size.
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 1)  # Single output for binary classification
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, batch):
        """Forward pass to predict whether the next sentence is the correct one.

        Args:
            batch (torch.Tensor): Output embeddings from BERT model (only [CLS] token).

        Returns:
            torch.Tensor: Probability of the sentence being the correct next sentence.
        """
        logits = self.linear(batch[:, 0])  # Use [CLS] token
        return self.sigmoid(logits)  # Sigmoid output for probability of "is_next"


class MaskedLanguageModel(torch.nn.Module):
    """Masked Language Model that predicts the original token from a masked input sequence.

    Args:
        hidden (int): BERT model output size.
        vocab_size (int): Size of the vocabulary.
    """
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for masked language model.

        Args:
            batch (torch.Tensor): Input tensor with masked tokens.

        Returns:
            torch.Tensor: Output logits after applying linear transformation and softmax.
        """
        return self.softmax(self.linear(batch))


class BERTLM(L.LightningModule):
    """BERT Language Model for pretraining: Next Sentence Prediction + Masked Language Model

    Args:
        config (BertConfig): Configuration object that contains model parameters.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = Bert(config)
        self.next_sentence = NextSentencePrediction(config.d_model)
        self.mask_lm = MaskedLanguageModel(config.d_model, config.vocab_size)

    def training_step(self, batch, segment_label):
        """Training step to compute the loss for both NSP and Masked LM tasks.

        Args:
            batch (torch.Tensor): Input token indices.
            segment_label (torch.Tensor): Segment indices.

        Returns:
            tuple: Tuple of Next Sentence Prediction and Masked LM outputs.
        """
        batch = self.bert(batch, segment_label)
        return self.next_sentence(batch), self.mask_lm(batch)

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler for training.

        Returns:
            ScheduledOptim: Optimizer wrapped with learning rate scheduling.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return ScheduledOptim(optimizer, self.config.d_model, self.config.warmup_steps)
