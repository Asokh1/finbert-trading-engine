import os
import math
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from dotenv import load_dotenv
from api_utils import fetch_json

load_dotenv()

MODEL_DIR = 'models/finbert_renewable'
BASE_MODEL = 'ProsusAI/finbert'
FINNHUB_API = os.getenv('FINNHUB_API_KEY')
device = torch.device('cpu')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def predict_positivity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    # FinBERT id2label: 0=positive, 1=negative, 2=neutral. Split neutral mass evenly so
    # a neutral headline pulls the score toward 0.5 instead of being ignored outright.
    return float(probs[0] + 0.5 * probs[2])

def score_articles(articles, model, tokenizer):
    return [predict_positivity(article.get('headline', ''), model, tokenizer) for article in articles]

def sample_articles(articles, max_count=40):
    # Sample evenly to prevent the model from only reading a single day's news
    step = max(1, len(articles) // max_count)
    return articles[::step][:max_count]

def fetch_finnhub_news(symbol, from_date, to_date):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API}'
    return fetch_json(url, f"{symbol} on {to_date}")

def calculate_historical_score(articles, positivities, target_date, half_life_days):
    total_weight = 0
    weighted_sentiment = 0

    for article, positivity in zip(articles, positivities):
        timestamp = article.get('datetime', 0)
        article_date = datetime.fromtimestamp(timestamp)

        # Only look at articles BEFORE our target date
        if article_date >= target_date:
            continue

        days_ago = (target_date - article_date).days
        # Only look at the 30 days leading up to the target date
        if days_ago > 30 or days_ago < 0:
            continue

        weight = math.pow(0.5, days_ago / half_life_days)

        weighted_sentiment += positivity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return weighted_sentiment / total_weight

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()

def calculate_trade_return(price_in, price_out, is_bullish, cost_pct):
    # cost_pct hits us going in and coming out, roughly commission + slippage
    if is_bullish:
        effective_in = price_in * (1 + cost_pct)
        effective_out = price_out * (1 - cost_pct)
        return (effective_out - effective_in) / effective_in
    else: # Bearish / Shorting
        effective_in = price_in * (1 - cost_pct)
        effective_out = price_out * (1 + cost_pct)
        return (effective_in - effective_out) / effective_in
