import torch
from transformers.models.bert.modeling_bert import (
    BertModel as HF_BertModel,
    BertEmbeddings,
    BertLayer,
    BertAttention,
)
from transformers import AutoTokenizer
from crammers.models import (
    BertVariantConfig,
    BertLM,
    BertConfig,
    BertEmbedding,
    BertEncoderLayer,
    MultiHeadedAttention,
)

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    text = "hello my name is shahrukh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    config: BertConfig = BertVariantConfig.BASE_UNCASED.value
    bert_model = BertLM(config)
    bert_model = bert_model.load_pretrained_bert(model_name)
    bert_model = bert_model.eval()
    bert_embeddings: BertEmbedding = bert_model.encoder.embeddings
    bert_encoder_layer: MultiHeadedAttention = bert_model.encoder.encoder_blocks[
        0
    ].self_multihead

    hf_bert_model: HF_BertModel = HF_BertModel.from_pretrained(model_name)
    hf_bert_model = hf_bert_model.eval()
    hf_bert_embeddings: BertEmbeddings = hf_bert_model.embeddings
    hf_bert_encoder_layer: BertAttention = hf_bert_model.encoder.layer[0].attention
    filtered_dict = {
        k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
    }

    # print(tokens['input_ids'].shape)
    hf_embeddings = hf_bert_embeddings(**filtered_dict)
    custom_embeddings = bert_embeddings(**filtered_dict)
    contextualized_hf_embeddings, hattn = hf_bert_encoder_layer(hf_embeddings)
    contextualized_custom_embeddings, attn = bert_encoder_layer(custom_embeddings, custom_embeddings, custom_embeddings)
    # print(len(contextualized_hf_embeddings))
    # print(contextualized_hf_embeddings[1], contextualized_custom_embeddings)
    # print(contextualized_custom_embeddings.shape)
    # print(contextualized_hf_embeddings)
    # print(contextualized_custom_embeddings)
    # print(contextualized_hf_embeddings)

    # print(
    #     torch.allclose(
    #         contextualized_hf_embeddings[0], contextualized_custom_embeddings, atol=1e-6
    #     )
    # )
    print(
        torch.max(
            torch.abs(
                contextualized_hf_embeddings - contextualized_custom_embeddings
            )
        )
    )
