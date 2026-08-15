from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from signal_engine import (
    load_model, score_articles, sample_articles, fetch_finnhub_news,
    calculate_historical_score, calculate_atr,
)
from trade_engine import simulate_trade, compute_performance_metrics, fetch_benchmark_equity, plot_equity_curve

STOCKS = ['AMZN', 'TSLA', 'AAPL', 'MSFT', 'GOOGL']

ATR_MULTIPLIER = 2 # stop = 2x ATR
TAKE_PROFIT_MULTIPLIER = 3 # target = 3x ATR, so we're risking 1 to make 1.5
TRANSACTION_COST_PCT = 0.001 # ~10bps per leg for commission/slippage
RISK_PER_TRADE_PCT = 0.01 # risk about 1% of capital per trade
MAX_POSITION_WEIGHT = 3.0 # cap so a super tight stop doesn't over-leverage us
HOLDING_PERIOD_DAYS = 7

# Threshold picked from the empirical momentum distribution (std ~0.007 under the
# 3-class model) to land near the ~11% signal rate the strategy was tuned around,
# rather than the old 0.02 cutoff which only caught ~2%.
MOMENTUM_BULLISH_THRESHOLD = 0.01
MOMENTUM_BEARISH_THRESHOLD = -0.01


def get_test_window():
    # Finnhub's free tier only serves company news for roughly the trailing year, so
    # 355 days is close to the max history we can pull without silently getting empty
    # results near the boundary.
    end_date = datetime.now() - timedelta(days=10)
    start_date = end_date - timedelta(days=355)
    test_dates = pd.date_range(start=start_date, end=end_date, freq='W')
    return start_date, end_date, test_dates


def run_backtest():
    print("Initializing Backtester with Dynamic ATR Stop-Loss...")
    model, tokenizer = load_model()

    start_date, end_date, test_dates = get_test_window()

    total_trades = 0
    winning_trades = 0
    trade_records = []

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
            news = fetch_finnhub_news(symbol, from_date, to_date)

            if not news:
                continue

            sampled_news = sample_articles(news, max_count=40)

            # score each article once and reuse it below, instead of scoring it twice
            sampled_positivity = score_articles(sampled_news, model, tokenizer)

            # 3. Calculate historical momentum
            short_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 3)
            long_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 14)
            momentum = short_term - long_term

            # 4. Define our Signal Strategy
            is_bullish = momentum > MOMENTUM_BULLISH_THRESHOLD
            is_bearish = momentum < MOMENTUM_BEARISH_THRESHOLD

            if not is_bullish and not is_bearish:
                continue # Skip if no strong signal

            signal_text = "BULLISH (BUY)" if is_bullish else "BEARISH (SHORT)"

            try:
                trade = simulate_trade(
                    stock_data, target_date, is_bullish, is_bearish,
                    ATR_MULTIPLIER, TAKE_PROFIT_MULTIPLIER, TRANSACTION_COST_PCT,
                    RISK_PER_TRADE_PCT, MAX_POSITION_WEIGHT, HOLDING_PERIOD_DAYS,
                )
                if trade is None:
                    continue

                total_trades += 1
                if trade['trade_return'] > 0:
                    winning_trades += 1

                trade['symbol'] = symbol
                trade_records.append(trade)

                # marker so we know how it closed
                status_marker = {'stop': '[STOPPED OUT]', 'target': '[TARGET HIT]', 'time': ''}[trade['exit_reason']]
                print(f"{target_date.strftime('%Y-%m-%d'):<12} {symbol:<6} {signal_text:<22} "
                      f"${trade['price_in']:<9.2f} ${trade['price_out']:<9.2f} {trade['position_weight']:<7.2f}x "
                      f"{trade['trade_return']*100:>6.2f}%  {status_marker}")

            except Exception as e:
                print(f"Skipping {symbol} on {target_date.strftime('%Y-%m-%d')}: {e}")

    print("=" * 85)
    print(f"Total Trades Taken:  {total_trades}")
    print(f"Winning Trades:      {winning_trades}")
    if total_trades > 0:
        print(f"Win Rate:            {(winning_trades/total_trades)*100:.1f}%")

        metrics = compute_performance_metrics(trade_records, start_date, end_date)
        print(f"Cumulative PnL:      {metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe']:.2f}")
        print(f"Sortino Ratio:       {metrics['sortino']:.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")

        benchmark_dates, benchmark_equity = fetch_benchmark_equity(start_date, end_date)
        if benchmark_equity is not None:
            benchmark_return = benchmark_equity.iloc[-1] - 1.0
            print(f"Benchmark (SPY B&H): {benchmark_return*100:.2f}%")
            print(f"Alpha vs Benchmark:  {(metrics['total_return'] - benchmark_return)*100:.2f}%")

        curves = [{'label': 'Strategy', 'dates': metrics['dates'], 'equity': metrics['equity_curve'], 'style': {'linewidth': 2}}]
        if benchmark_equity is not None:
            curves.append({'label': 'SPY Buy & Hold', 'dates': benchmark_dates, 'equity': benchmark_equity, 'style': {'linewidth': 1.5, 'alpha': 0.8}})
        plot_equity_curve(curves)

if __name__ == '__main__':
    run_backtest()
