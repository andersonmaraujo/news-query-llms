# app.py

from fetch_news import fetch_latest_news, store_news_in_dataframe
from vector_db import initialize_pinecone, store_embeddings
from query_answer import retrieve_relevant_news, generate_summary_response

def main():
    # Fetch news
    news = fetch_latest_news('technology')
    news_df = store_news_in_dataframe(news)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Store embeddings
    articles = news_df.to_dict(orient='records')
    store_embeddings(index, articles)
    
    # Example query
    query = "What most have impacted NVIDIA stocks recently?"
    relevant_news = retrieve_relevant_news(query)
    summary = generate_summary_response(query, relevant_news)
    print(summary)

if __name__ == "__main__":
    main()
