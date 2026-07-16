import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import requests
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
import os
import math
import statistics
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv()

MODEL_DIR = 'models/finbert_renewable'
BASE_MODEL = 'ProsusAI/finbert'
FINNHUB_API = os.getenv('FINNHUB_API_KEY')

STOCKS = ['AMZN', 'TSLA', 'AAPL', 'MSFT', 'GOOGL']
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

def score_articles(articles, model, tokenizer):
    return [predict_positivity(article.get('headline', ''), model, tokenizer) for article in articles]

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

def compute_performance_metrics(trade_records, start_date):
    sorted_trades = sorted(trade_records, key=lambda t: t['date'])
    weighted_returns = [t['weighted_return'] for t in sorted_trades]
    dates = [start_date] + [t['date'] for t in sorted_trades]

    # compounded equity curve, needed for max drawdown and the plot
    equity_curve = [1.0]
    for r in weighted_returns:
        equity_curve.append(equity_curve[-1] * (1 + r))

    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, (value - peak) / peak)

    n = len(weighted_returns)
    mean_return = statistics.mean(weighted_returns)
    std_return = statistics.pstdev(weighted_returns) if n > 1 else 0.0

    # annualize using the trade frequency actually observed over the backtest window
    span_days = max((dates[-1] - start_date).days, 1)
    periods_per_year = n * 365 / span_days

    sharpe = (mean_return / std_return) * math.sqrt(periods_per_year) if std_return > 0 else 0.0

    downside_returns = [r for r in weighted_returns if r < 0]
    downside_std = statistics.pstdev(downside_returns) if len(downside_returns) > 1 else 0.0
    sortino = (mean_return / downside_std) * math.sqrt(periods_per_year) if downside_std > 0 else 0.0

    return {
        'dates': dates,
        'equity_curve': equity_curve,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'total_return': equity_curve[-1] - 1.0,
    }


def fetch_benchmark_equity(start_date, end_date, symbol='SPY'):
    benchmark = yf.Ticker(symbol).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if benchmark.empty:
        return None, None
    benchmark.index = benchmark.index.tz_localize(None)
    equity = benchmark['Close'] / benchmark['Close'].iloc[0]
    return benchmark.index, equity


def plot_equity_curve(portfolio_dates, portfolio_equity, benchmark_dates, benchmark_equity, benchmark_symbol, output_path='equity_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.step(portfolio_dates, portfolio_equity, where='post', label='Strategy', linewidth=2)
    if benchmark_equity is not None:
        plt.plot(benchmark_dates, benchmark_equity, label=f'{benchmark_symbol} Buy & Hold', linewidth=1.5, alpha=0.8)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Strategy Equity Curve vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Equity curve saved to {output_path}")


def run_backtest():
    print("Initializing Backtester with Dynamic ATR Stop-Loss...")
    model, tokenizer = load_model()
    
    # Test signals from 210 days ago up to 10 days ago
    end_date = datetime.now() - timedelta(days=10)
    start_date = end_date - timedelta(days=200)
    
    # Generate weekly test dates
    test_dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    total_trades = 0
    winning_trades = 0
    trade_records = []
    ATR_MULTIPLIER = 2 # stop = 2x ATR
    TAKE_PROFIT_MULTIPLIER = 3 # target = 3x ATR, so we're risking 1 to make 1.5
    TRANSACTION_COST_PCT = 0.001 # ~10bps per leg for commission/slippage
    RISK_PER_TRADE_PCT = 0.01 # risk about 1% of capital per trade
    MAX_POSITION_WEIGHT = 3.0 # cap so a super tight stop doesn't over-leverage us

    print(f"\n{'DATE':<12} {'SYM':<6} {'SIGNAL':<22} {'PRICE IN':<10} {'PRICE OUT':<10} {'SIZE':<8} {'RETURN'}")
    print("=" * 85)

    for symbol in STOCKS:
        # 1. Fetch historical prices once per stock using yf.Ticker for safety
        ticker = yf.Ticker(symbol)
        # extra 30-day buffer so the ATR has warmed up by start_date
        stock_data = ticker.history(start=(start_date - timedelta(days=30)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))

        if stock_data.empty:
            continue

        # Remove timezone awareness to match our target_date so they align perfectly
        stock_data.index = stock_data.index.tz_localize(None)
        stock_data['ATR'] = calculate_atr(stock_data, period=14)

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

            # score each article once and reuse it below, instead of scoring it twice
            sampled_positivity = score_articles(sampled_news, model, tokenizer)

            # 3. Calculate historical momentum
            short_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 3)
            long_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 14)
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

                current_atr = stock_data['ATR'].iloc[price_in_idx]
                if pd.isna(current_atr):
                    continue # not enough history yet for a 14-day ATR

                # stop and target are both set once at entry, based on that day's ATR
                if is_bullish:
                    stop_loss_price = price_in - (ATR_MULTIPLIER * current_atr)
                    take_profit_price = price_in + (TAKE_PROFIT_MULTIPLIER * current_atr)
                else: # shorting - stop sits above entry, target sits below
                    stop_loss_price = price_in + (ATR_MULTIPLIER * current_atr)
                    take_profit_price = price_in - (TAKE_PROFIT_MULTIPLIER * current_atr)

                # wider stop = smaller size, so every trade risks about the same amount
                stop_distance_pct = (ATR_MULTIPLIER * current_atr) / price_in
                position_weight = min(RISK_PER_TRADE_PCT / stop_distance_pct, MAX_POSITION_WEIGHT)

                # Get the next 7 days of price data
                exit_date = actual_entry_date + timedelta(days=7)
                holding_period_data = stock_data.loc[actual_entry_date + timedelta(days=1) : exit_date]

                trade_return = 0.0
                price_out = price_in
                stopped_out = False
                hit_target = False

                # Check price day by day
                for current_date, row in holding_period_data.iterrows():
                    # stop-loss first, in case a gap day blows through both levels
                    if is_bullish and row['Low'] < stop_loss_price:
                        price_out = stop_loss_price
                        trade_return = calculate_trade_return(price_in, price_out, is_bullish, TRANSACTION_COST_PCT)
                        stopped_out = True
                        break # Exit the trade instantly!
                    elif is_bearish and row['High'] > stop_loss_price:
                        price_out = stop_loss_price
                        trade_return = calculate_trade_return(price_in, price_out, is_bullish, TRANSACTION_COST_PCT)
                        stopped_out = True
                        break # Exit the trade instantly!

                    # then check if we hit the target
                    if is_bullish and row['High'] > take_profit_price:
                        price_out = take_profit_price
                        trade_return = calculate_trade_return(price_in, price_out, is_bullish, TRANSACTION_COST_PCT)
                        hit_target = True
                        break
                    elif is_bearish and row['Low'] < take_profit_price:
                        price_out = take_profit_price
                        trade_return = calculate_trade_return(price_in, price_out, is_bullish, TRANSACTION_COST_PCT)
                        hit_target = True
                        break

                # survived the week without hitting either level, close normally
                if not stopped_out and not hit_target and not holding_period_data.empty:
                    price_out = float(holding_period_data.iloc[-1]['Close'])
                    trade_return = calculate_trade_return(price_in, price_out, is_bullish, TRANSACTION_COST_PCT)

                total_trades += 1
                if trade_return > 0:
                    winning_trades += 1

                trade_records.append({
                    'date': actual_entry_date,
                    'symbol': symbol,
                    'weighted_return': trade_return * position_weight,
                })

                # marker so we know how it closed
                if stopped_out:
                    status_marker = "[STOPPED OUT]"
                elif hit_target:
                    status_marker = "[TARGET HIT]"
                else:
                    status_marker = ""
                print(f"{target_date.strftime('%Y-%m-%d'):<12} {symbol:<6} {signal_text:<22} ${price_in:<9.2f} ${price_out:<9.2f} {position_weight:<7.2f}x {trade_return*100:>6.2f}%  {status_marker}")
                
            except Exception as e:
                pass # Skip if market was closed or prices missing

    print("=" * 85)
    print(f"Total Trades Taken:  {total_trades}")
    print(f"Winning Trades:      {winning_trades}")
    if total_trades > 0:
        print(f"Win Rate:            {(winning_trades/total_trades)*100:.1f}%")

        metrics = compute_performance_metrics(trade_records, start_date)
        print(f"Cumulative PnL:      {metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe']:.2f}")
        print(f"Sortino Ratio:       {metrics['sortino']:.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")

        benchmark_dates, benchmark_equity = fetch_benchmark_equity(start_date, end_date)
        if benchmark_equity is not None:
            benchmark_return = benchmark_equity.iloc[-1] - 1.0
            print(f"Benchmark (SPY B&H): {benchmark_return*100:.2f}%")
            print(f"Alpha vs Benchmark:  {(metrics['total_return'] - benchmark_return)*100:.2f}%")

        plot_equity_curve(metrics['dates'], metrics['equity_curve'], benchmark_dates, benchmark_equity, 'SPY')

if __name__ == '__main__':
    run_backtest()