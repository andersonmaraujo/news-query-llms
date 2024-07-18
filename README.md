# Startup News Search Application

This application collects the latest startup news from [NewsData.io](https://newsdata.io/) and allows users to ask questions about the latest news. The application uses a vector database to store news embeddings and provides relevant search results based on user queries.

## Features

- Fetches the latest startup news articles.
- Generates embeddings for news articles using BERT.
- Stores embeddings in a Pinecone vector database.
- Allows users to input queries and retrieve relevant news articles.
- Provides an interactive web interface using Streamlit.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- [NewsData.io API Key](https://newsdata.io/)
- [Pinecone API Key](https://www.pinecone.io/)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/startup-news-search.git
   cd startup-news-search
