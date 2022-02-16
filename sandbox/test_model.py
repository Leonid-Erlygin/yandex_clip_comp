import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("../baseline/text_models/rubert-tiny")
model = AutoModel.from_pretrained("../baseline/text_models/rubert-tiny")
model.cuda()  # uncomment it if you have a GPU

def predict_outputs(model, t):
    model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


texts = ['привет мир', 'русские вперёд папа']
ts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

input_ids = ts['input_ids'].to(model.device)
token_type_ids = ts['token_type_ids'].to(model.device)
attention_mask = ts['attention_mask'].to(model.device)

model_output = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
embeddings = model_output.last_hidden_state[:, 0, :]
embeddings = torch.nn.functional.normalize(embeddings)
print(embeddings.shape)
#model_outputs = torch.cat([predict_outputs(model, tokenizer(text, padding=True, truncation=True, return_tensors='pt')) for text in texts])


