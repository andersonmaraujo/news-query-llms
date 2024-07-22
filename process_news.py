# process_news.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Embed text using the model
def embed_text(text):
    if not text:
        return np.zeros(384)  # Return a zero vector for empty text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Example usage
if __name__ == "__main__":
    example_text = "NVIDIA reported higher-than-expected earnings for the last quarter."
    embedding = embed_text(example_text)
    print(embedding)
