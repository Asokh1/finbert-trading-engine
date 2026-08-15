from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from api_utils import fetch_json
from signal_engine import load_model, score_articles, sample_articles, calculate_historical_score

load_dotenv()

FINNHUB_API = os.getenv('FINNHUB_API_KEY')

STOCKS = ['MU']

def get_historical_news(symbol, days=30):
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={today}&token={FINNHUB_API}'
    return fetch_json(url, symbol)

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
            
        sampled_news = sample_articles(news, max_count=40)
        sampled_positivity = score_articles(sampled_news, model, tokenizer)

        now = datetime.now()
        short_term_score = calculate_historical_score(sampled_news, sampled_positivity, now, half_life_days=3)
        long_term_score = calculate_historical_score(sampled_news, sampled_positivity, now, half_life_days=14)

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