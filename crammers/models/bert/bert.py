from dataclasses import dataclass

from transformers import BertModel

from loguru import logger
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
        type_vocab_size (int): Number of segment types.
        embed_size (int): Dimension of the embeddings.
        seq_len (int): Maximum sequence length for the input.
        heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        feed_forward_hidden (int): Hidden layer size of the feedforward network.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attention_probs_dropout_prob (float): Dropout probability for attention layers.
        classifier_dropout (float): Dropout probability for the classifier layer.
        pad_token_id (int): Token ID for padding.
        n_layers (int): Number of layers in the encoder.
        layer_norm_eps (float): Epsilon value for layer normalization.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        warmup_steps (int): Number of steps for the learning rate warmup.
    """

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=3,
        embed_size=768,
        seq_len=512,
        heads=12,
        d_model=768,
        feed_forward_hidden=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=None,
        pad_token_id=0,
        n_layers=12,
        layer_norm_eps=1e-12,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=4000,
    ):
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.heads = heads
        self.d_model = d_model
        self.feed_forward_hidden = feed_forward_hidden
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers
        self.layer_norm_eps = layer_norm_eps

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps


class BertEmbedding(torch.nn.Module):
    """BERT embedding layer that generates token, positional, and segment embeddings.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embed_size = config.embed_size
        self.token = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=config.pad_token_id
        )
        self.segment = nn.Embedding(config.type_vocab_size, config.embed_size)
        self.position = nn.Embedding(config.seq_len, config.embed_size)
        self.layer_norm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.seq_len).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Generates embeddings by adding token, positional, and segment embeddings.

        Args:
            input_ids (torch.Tensor): Tensor of input token indices.
            token_type_ids (torch.Tensor): Tensor of segment indices.

        Returns:
            torch.Tensor: Output embeddings after adding token, position, and segment embeddings.
        """
        embeddings = (
            self.token(input_ids)
            + self.position(self.position_ids[:, : input_ids.size(1)])
            + self.segment(token_type_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        self.query = torch.nn.Linear(config.d_model, config.d_model)
        self.key = torch.nn.Linear(config.d_model, config.d_model)
        self.value = torch.nn.Linear(config.d_model, config.d_model)

        self.dense = torch.nn.Linear(config.d_model, config.d_model)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """Forward pass for multi-headed attention layer.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention.
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
        if mask:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, value)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )

        hidden_states = self.dense(context)
        hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.layer_norm(hidden_states + context)
        return hidden_states, context


class FFN(nn.Module):
    """Feed-forward network for the BERT encoder layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(FFN, self).__init__()
        self.intermediate_dim = config.feed_forward_hidden
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        self.layernorm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_multihead = MultiHeadedAttention(config)
        self.feed_forward = FFN(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for a single encoder layer.

        Args:
            embeddings (torch.Tensor): Input embeddings.
            mask (torch.Tensor): Mask tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output embeddings after applying multi-head attention and feed-forward network.
        """
        interacted = self.self_multihead(embeddings, embeddings, embeddings, mask)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        return self.layernorm(feed_forward_out + interacted)


class Bert(nn.Module):
    """Vanilla BERT model implementation with multiple transformer layers.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.embeddings = BertEmbedding(config)
        self.encoder_blocks = torch.nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids, token_type_ids):
        """Forward pass for the entire BERT model.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment labels (e.g., sentence A/B).

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
        embeddings = self.embeddings(input_ids, token_type_ids)

        for encoder in self.encoder_blocks:
            embeddings = encoder(embeddings, mask)
        return embeddings


class NextSentencePrediction(torch.nn.Module):
    """2-class classification model: is_next, is_not_next

    Args:
        hidden (int): BERT model output size.
    """

    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(
            hidden, 1
        )  # Single output for binary classification
        self.sigmoid = (
            torch.nn.Sigmoid()
        )  # Sigmoid activation for binary classification

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


class BertLM(L.LightningModule):
    """BERT Language Model for pretraining: Next Sentence Prediction + Masked Language Model

    Args:
        config (BertConfig): Configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.encoder = Bert(config)
        self.next_sentence = NextSentencePrediction(config.d_model)
        self.mask_lm = MaskedLanguageModel(config.d_model, config.vocab_size)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the entire BERT model.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment labels (e.g., sentence A/B).

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        return self.encoder(input_ids, token_type_ids)

    def training_step(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training step to compute the loss for both NSP and Masked LM tasks.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment indices.

        Returns:
            tuple: Tuple of Next Sentence Prediction and Masked LM outputs.
        """
        embeddings = self(input_ids, token_type_ids)
        return self.next_sentence(embeddings), self.mask_lm(embeddings)

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

    def load_pretrained_bert(self, model_name: str = "bert-base-uncased"):
        # sourcery skip: extract-method
        """
        Load pretrained weights from HuggingFace Transformers BERT model into custom BERT implementation.

        Args:
            model_name (str): Name of pretrained model to load from HuggingFace

        Returns:
            BERTLM: Custom model with loaded pretrained weights
            None: If transformers is not installed
        """
        try:
            # Load pretrained model
            pretrained = BertModel.from_pretrained(model_name)

            # Map embeddings
            self.encoder.embeddings.token.weight.data = (
                pretrained.embeddings.word_embeddings.weight.data
            )
            self.encoder.embeddings.position.weight.data = (
                pretrained.embeddings.position_embeddings.weight.data
            )
            self.encoder.embeddings.segment.weight.data = (
                pretrained.embeddings.token_type_embeddings.weight.data
            )
            self.encoder.embeddings.layer_norm.weight.data = (
                pretrained.embeddings.LayerNorm.weight.data
            )
            self.encoder.embeddings.layer_norm.bias.data = (
                pretrained.embeddings.LayerNorm.bias.data
            )
            # Map encoder layers
            for custom_layer, pretrained_layer in zip(
                self.encoder.encoder_blocks, pretrained.encoder.layer
            ):
                # Self attention weights
                custom_layer.self_multihead.query.weight.data = (
                    pretrained_layer.attention.self.query.weight.data
                )
                custom_layer.self_multihead.query.bias.data = (
                    pretrained_layer.attention.self.query.bias.data
                )
                custom_layer.self_multihead.key.weight.data = (
                    pretrained_layer.attention.self.key.weight.data
                )
                custom_layer.self_multihead.key.bias.data = (
                    pretrained_layer.attention.self.key.bias.data
                )
                custom_layer.self_multihead.value.weight.data = (
                    pretrained_layer.attention.self.value.weight.data
                )
                custom_layer.self_multihead.value.bias.data = (
                    pretrained_layer.attention.self.value.bias.data
                )
                custom_layer.self_multihead.dense.weight.data = (
                    pretrained_layer.attention.output.dense.weight.data
                )
                custom_layer.self_multihead.dense.bias.data = (
                    pretrained_layer.attention.output.dense.bias.data
                )
                custom_layer.self_multihead.layer_norm.weight.data = (
                    pretrained_layer.attention.output.LayerNorm.weight.data
                )
                custom_layer.self_multihead.layer_norm.bias.data = (
                    pretrained_layer.attention.output.LayerNorm.bias.data
                )

                # Layer norm weights
                custom_layer.layernorm.weight.data = (
                    pretrained_layer.attention.output.LayerNorm.weight.data
                )
                custom_layer.layernorm.bias.data = (
                    pretrained_layer.attention.output.LayerNorm.bias.data
                )

                # Feed forward weights
                custom_layer.feed_forward.fc1.weight.data = (
                    pretrained_layer.intermediate.dense.weight.data
                )
                custom_layer.feed_forward.fc1.bias.data = (
                    pretrained_layer.intermediate.dense.bias.data
                )
                custom_layer.feed_forward.fc2.weight.data = (
                    pretrained_layer.output.dense.weight.data
                )
                custom_layer.feed_forward.fc2.bias.data = (
                    pretrained_layer.output.dense.bias.data
                )

            logger.info(
                f"Successfully loaded weights from pretrained model: {model_name}"
            )
            return self

        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            return None
