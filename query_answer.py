# query_answer.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from process_news import embed_text
from vector_db import initialize_pinecone

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_query(query):
    if not query:
        return np.zeros(384).tolist()  # Return a zero vector for empty query
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

def retrieve_relevant_news(query):
    index = initialize_pinecone()
    query_embedding = embed_query(query)
    if np.count_nonzero(query_embedding) == 0:
        return np.zeros(384).tolist()  # Return a zero vector for empty query
    
    # Ensure the query embedding values are in float32 format and within a valid range
    query_embedding = np.clip(np.array(query_embedding, dtype=np.float32), -1.0, 1.0).tolist()
    
    # Convert numpy array to list of lists for Pinecone
    results = index.query(vector=[query_embedding], top_k=5)
    return results['matches']


def generate_summary_response(query, articles):
    summary = f"The most significant factors impacting {query} include:\n\n"
    print(articles)
    for article in articles:
        summary += f"- {article['title']}: {article['content'][:200]}...\n"
        summary += f"  [Read more]({article['url']})\n\n"
    return summary

# Example usage
if __name__ == "__main__":
    example_query = "What most have impacted NVIDIA stocks recently?"
    relevant_news = retrieve_relevant_news(example_query)
    summary = generate_summary_response(example_query, relevant_news)
    print(summary)
