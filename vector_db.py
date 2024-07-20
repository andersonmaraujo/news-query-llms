# vector_db.py

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from process_news import embed_text

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'news-articles'
    return pc.Index(index_name)

def store_embeddings(index, articles):
    for article in articles:
        embedding = embed_text(article['title'] + " " + article['content'])
        index.upsert([(article['article_id'], embedding.tolist())])

# Example usage
if __name__ == "__main__":
    index = initialize_pinecone()
    articles = [
        {"id": "1", "title": "NVIDIA earnings report", "content": "NVIDIA reported higher-than-expected earnings for the last quarter."},
        # Add more articles
    ]
    store_embeddings(index, articles)
    print(index.describe_index_stats())
