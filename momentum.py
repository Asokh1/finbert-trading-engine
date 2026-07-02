import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import math

load_dotenv()

MODEL_DIR = 'models/finbert_renewable'
BASE_MODEL = 'ProsusAI/finbert'
FINNHUB_API = os.getenv('FINNHUB_API_KEY')

STOCKS = ['MU']
device = torch.device('cpu')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def get_historical_news(symbol, days=30):
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={today}&token={FINNHUB_API}'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def predict_positivity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    return probs[1]

def calculate_time_decayed_score(articles, model, tokenizer, half_life_days):
    if not articles:
        return 0.5 

    total_weight = 0
    weighted_sentiment = 0
    now = datetime.now()

    for article in articles:
        headline = article.get('headline', '')
        if not headline:
            continue
            
        timestamp = article.get('datetime', 0)
        article_date = datetime.fromtimestamp(timestamp)
        days_ago = (now - article_date).days
        if days_ago < 0: days_ago = 0
        
        weight = math.pow(0.5, days_ago / half_life_days)
        
        positivity = predict_positivity(headline, model, tokenizer)
        
        weighted_sentiment += positivity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
        
    return weighted_sentiment / total_weight

def analyze_momentum():
    print("Loading model...")
    model, tokenizer = load_model()
    
    print(f"\n{'SYMBOL':<8} {'SHORT-TERM (7d)':<17} {'LONG-TERM (30d)':<17} {'MACD MOMENTUM':<15} {'SIGNAL'}")
    print("=" * 85)
    
    for symbol in STOCKS:
        news = get_historical_news(symbol, days=30)
        
        if not news:
            print(f"{symbol:<8} No Data")
            continue
            
        step = max(1, len(news) // 40)
        sampled_news = news[::step][:40]
            
        short_term_score = calculate_time_decayed_score(sampled_news, model, tokenizer, half_life_days=3)
        
        long_term_score = calculate_time_decayed_score(sampled_news, model, tokenizer, half_life_days=14)
        
        momentum = short_term_score - long_term_score
        
        if momentum > 0.05:
            signal = "BULLISH ACCELERATION"
        elif momentum < -0.05:
            signal = "BEARISH REVERSAL"
        elif momentum > 0:
            signal = "SLIGHT UPTREND"
        else:
            signal = "SLIGHT DOWNTREND"
            
        print(f"{symbol:<8} {short_term_score:<17.4f} {long_term_score:<17.4f} {momentum:<15.4f} {signal}")

if __name__ == '__main__':
    analyze_momentum()