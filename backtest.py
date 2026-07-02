import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import requests
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
import os
import math
import pandas as pd

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

def predict_positivity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    return probs[1]

def calculate_historical_score(articles, target_date, model, tokenizer, half_life_days):
    total_weight = 0
    weighted_sentiment = 0

    for article in articles:
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
        positivity = predict_positivity(article.get('headline', ''), model, tokenizer)
        
        weighted_sentiment += positivity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return weighted_sentiment / total_weight

def run_backtest():
    print("Initializing Backtester with 5% Stop-Loss...")
    model, tokenizer = load_model()
    
    # Test signals from 60 days ago up to 10 days ago
    end_date = datetime.now() - timedelta(days=10)
    start_date = end_date - timedelta(days=50)
    
    # Generate weekly test dates
    test_dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    total_trades = 0
    winning_trades = 0
    portfolio_return = 0.0
    stop_loss_limit = -0.05 # Automatically exit if we lose 5%

    print(f"\n{'DATE':<12} {'SYM':<6} {'SIGNAL':<22} {'PRICE IN':<10} {'PRICE OUT':<10} {'RETURN'}")
    print("=" * 85)

    for symbol in STOCKS:
        # 1. Fetch historical prices once per stock using yf.Ticker for safety
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        
        if stock_data.empty:
            continue
            
        # Remove timezone awareness to match our target_date so they align perfectly
        stock_data.index = stock_data.index.tz_localize(None)

        for target_date in test_dates:
            # 2. Fetch news specifically for the 30 days prior to THIS test date
            to_date = target_date.strftime('%Y-%m-%d')
            from_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API}'
            
            try:
                r = requests.get(url, timeout=5)
                news = r.json() if r.status_code == 200 else []
            except:
                news = []
                
            if not news:
                continue
                
            # Sample evenly to prevent AI from only reading a single day's news
            step = max(1, len(news) // 40)
            sampled_news = news[::step][:40]

            # 3. Calculate historical momentum
            short_term = calculate_historical_score(sampled_news, target_date, model, tokenizer, 3)
            long_term = calculate_historical_score(sampled_news, target_date, model, tokenizer, 14)
            momentum = short_term - long_term
            
            # 4. Define our Signal Strategy
            is_bullish = momentum > 0.02
            is_bearish = momentum < -0.02
            
            if not is_bullish and not is_bearish:
                continue # Skip if no strong signal
                
            signal_text = "BULLISH (BUY)" if is_bullish else "BEARISH (SHORT)"
            
            try:
                # Find entry day
                price_in_idx = stock_data.index.get_indexer([target_date], method='nearest')[0]
                actual_entry_date = stock_data.index[price_in_idx]
                price_in = float(stock_data['Close'].iloc[price_in_idx])
                
                # Get the next 7 days of price data
                exit_date = actual_entry_date + timedelta(days=7)
                holding_period_data = stock_data.loc[actual_entry_date + timedelta(days=1) : exit_date]
                
                trade_return = 0.0
                price_out = price_in
                stopped_out = False
                
                # Check price day by day
                for current_date, row in holding_period_data.iterrows():
                    current_price = row['Close']
                    
                    # Calculate live return
                    if is_bullish:
                        current_return = (current_price - price_in) / price_in
                    else: # Bearish / Shorting
                        current_return = (price_in - current_price) / price_in
                        
                    # Stop-loss check
                    if current_return <= stop_loss_limit:
                        trade_return = stop_loss_limit # We lock in the 5% loss
                        price_out = current_price
                        stopped_out = True
                        break # Exit the trade instantly!
                        
                # If we survived the 7 days without hitting the stop loss, close the trade normally
                if not stopped_out and not holding_period_data.empty:
                    price_out = float(holding_period_data.iloc[-1]['Close'])
                    if is_bullish:
                        trade_return = (price_out - price_in) / price_in
                    else:
                        trade_return = (price_in - price_out) / price_in
                
                portfolio_return += trade_return
                total_trades += 1
                if trade_return > 0:
                    winning_trades += 1
                    
                # Add a marker so we know which trades were stopped out
                status_marker = "[STOPPED OUT]" if stopped_out else ""
                print(f"{target_date.strftime('%Y-%m-%d'):<12} {symbol:<6} {signal_text:<22} ${price_in:<9.2f} ${price_out:<9.2f} {trade_return*100:>6.2f}%  {status_marker}")
                
            except Exception as e:
                pass # Skip if market was closed or prices missing

    print("=" * 85)
    print(f"Total Trades Taken:  {total_trades}")
    print(f"Winning Trades:      {winning_trades}")
    if total_trades > 0:
        print(f"Win Rate:            {(winning_trades/total_trades)*100:.1f}%")
        print(f"Cumulative PnL:      {portfolio_return*100:.2f}%")

if __name__ == '__main__':
    run_backtest()