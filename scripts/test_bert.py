import torch
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertModel as HF_BertModel,
)
from transformers import AutoTokenizer
from crammers.models import BertVariantConfig, BertLM, BertConfig, BertEmbedding

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    text = "hello my name is shahrukh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    config: BertConfig = BertVariantConfig.BASE_UNCASED.value
    bert_model = BertLM(config)
    bert_model = bert_model.load_pretrained_bert(model_name)
    bert_model = bert_model.eval()
    bert_embeddings: BertEmbedding = bert_model.bert.embeddings
    # print(bert_embeddings)

    hf_bert_model: HF_BertModel = HF_BertModel.from_pretrained(model_name)
    hf_bert_model = hf_bert_model.eval()
    hf_bert_embeddings: BertEmbeddings = hf_bert_model.embeddings
    print(tokens)
    filtered_dict = {
        k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
    }
    # print(tokens['input_ids'].shape)
    e1 = hf_bert_embeddings(**filtered_dict)
    e2 = bert_embeddings(**filtered_dict)
    # print()
    # print(bert_embeddings(**filtered_dict))
    # print(bert_embeddings.token(filtered_dict["input_ids"]))
    # print(hf_bert_embeddings.word_embeddings(filtered_dict["input_ids"]))
    # e1 = bert_embeddings.token(filtered_dict["input_ids"]) + bert_embeddings.segment(filtered_dict["token_type_ids"])
    e2 = hf_bert_embeddings.word_embeddings(
        filtered_dict["input_ids"]
    ) + hf_bert_embeddings.token_type_embeddings(filtered_dict["token_type_ids"])
    eqv = torch.allclose(e1, e2)
    print(eqv)
    print(
        torch.allclose(
            hf_bert_embeddings.LayerNorm.weight.data,
            bert_embeddings.layer_norm.weight.data,
        )
    )
    print(
        torch.allclose(
            hf_bert_embeddings.LayerNorm.bias.data, bert_embeddings.layer_norm.bias.data
        )
    )
