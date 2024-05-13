from transformers import BertConfig, BertModel
import torch
import time

# config = BertConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# config.hidden_size = 2048
# config.num_hidden_layers = 24
# config.num_attention_heads = 32
# config.max_position_embeddings = 512

model = BertModel.from_pretrained('/home/ubuntu/modelstorage/Model68B')

total_params = sum(p.numel() for p in model.parameters())
#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")