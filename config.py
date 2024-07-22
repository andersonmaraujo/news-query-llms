# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Newsdata.io API key
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
