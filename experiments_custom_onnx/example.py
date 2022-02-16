import numpy as np

from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer
from txtai.pipeline import HFOnnx

# Normalize logits using sigmoid function
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# Export to ONNX
onnx = HFOnnx()
model = onnx("distilbert-base-uncased-finetuned-sst-2-english", "text-classification")

# Start inference session
options = SessionOptions()
session = InferenceSession(model, options)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokens = tokenizer(["I am happy", "I am mad"], return_tensors="np")

# Print results
outputs = session.run(None, dict(tokens))
print(sigmoid(outputs[0]))


embeddings = onnx("sentence-transformers/paraphrase-MiniLM-L6-v2", "pooling", "embeddings.onnx", quantize=True)

print("工作")





