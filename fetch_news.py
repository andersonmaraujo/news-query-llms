# fetch_news.py

import requests
import pandas as pd
from config import NEWS_API_KEY

# Fetch latest news from newsdata.io
def fetch_latest_news(query, language='en', country='us'):
    url = 'https://newsdata.io/api/1/news'
    params = {
        'apikey': NEWS_API_KEY,
        'q': query,
        'language': language,
        'country': country
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['results']
    else:
        print(f"Failed to fetch news: {response.status_code}")
        return []

# Store news in a pandas dataframe
def store_news_in_dataframe(news_list):
    df = pd.DataFrame(news_list)
    return df

# Example usage
if __name__ == "__main__":
    news = fetch_latest_news('technology')
    news_df = store_news_in_dataframe(news)
    print(news_df.columns)
