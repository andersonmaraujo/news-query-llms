# app.py

from fetch_news import fetch_latest_news, store_news_in_dataframe
from vector_db import initialize_pinecone, store_embeddings

def main():
    # Fetch news
    news = fetch_latest_news('technology')
    news_df = store_news_in_dataframe(news)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Store embeddings
    articles = news_df.to_dict(orient='records')
    store_embeddings(index, articles)
    
    print("News articles have been fetched and stored.")

if __name__ == "__main__":
    main()
