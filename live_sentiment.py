import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

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

def get_stock_news(symbol):
    # Look back 7 days for recent news
    today = datetime.now().strftime('%Y-%m-%d')
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={last_week}&to={today}&token={FINNHUB_API}'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"API Error for {symbol}: {r.status_code}") # Helpful if you hit rate limits
    except:
        pass
    return []

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    label = 'POSITIVE' if probs[1] > probs[0] else 'NEGATIVE'
    confidence = max(probs)
    return label, confidence

def analyze_stocks():
    print("Loading trained model...")
    model, tokenizer = load_model()
    
    print(f"\n{'SYMBOL':<8} {'SENTIMENT':<12} {'CONFIDENCE':<12} {'LATEST NEWS':<50}")
    print("=" * 82)
    
    results = {}
    
    for symbol in STOCKS:
        news = get_stock_news(symbol)
        
        if not news:
            continue
        
        sentiments = []
        for article in news[:5]:
            headline = article.get('headline', '')
            if not headline:
                continue
            
            label, confidence = predict_sentiment(headline, model, tokenizer)
            sentiments.append({'label': label, 'confidence': confidence})
        
        if sentiments:
            avg_sentiment = 'POSITIVE' if sum(1 for s in sentiments if s['label']=='POSITIVE') > len(sentiments)/2 else 'NEGATIVE'
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            headline_preview = f"Based on avg of {len(sentiments)} recent articles"

            print(f"{symbol:<8} {avg_sentiment:<12} {avg_confidence:<12.4f} {headline_preview:<50}")
            
            results[symbol] = {
                'sentiment': avg_sentiment,
                'confidence': avg_confidence,
                'article_count': len(sentiments)
            }
    
    print("\n" + "=" * 82)
    print(f"\nAnalysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    positive = sum(1 for r in results.values() if r['sentiment'] == 'POSITIVE')
    negative = len(results) - positive
    
    print(f"Positive: {positive} | Negative: {negative}")

if __name__ == '__main__':
    analyze_stocks()