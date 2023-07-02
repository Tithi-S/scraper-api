from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from bs4 import BeautifulSoup
import requests
from transformers import pipeline



app = FastAPI()
sentiment_model = pipeline("sentiment-analysis")
# Endpoint for web scraping 

@app.get("/scrape-and-analyze/")
def scrape_website(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was not successful
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract the desired data from the website using web scraping
        scraped_data = extract_data(soup)
        # Perform sentiment analysis on the scraped data
        sentiment = analyze_sentiment(scraped_data)
        return {"scraped_data": scraped_data, "sentiment": sentiment}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail="Failed to fetch the website.")

# Endpoint for sentiment analysis
def extract_data(soup):
    # Add your web scraping code here to extract the desired data from the website
    # For simplicity, let's just return the raw HTML content
    return str(soup)

def analyze_sentiment(text):
    # Perform sentiment analysis on the text using the sentiment analysis model
    sentiment = sentiment_model(text)[0]
    return {"label": sentiment["label"], "score": sentiment["score"]}
