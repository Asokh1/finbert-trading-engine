from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf

import backtest
from signal_engine import (
    load_model, score_articles, sample_articles, fetch_finnhub_news,
    calculate_historical_score, calculate_atr,
)
from trade_engine import simulate_trade, compute_performance_metrics, fetch_benchmark_equity, plot_equity_curve

SHUFFLE_SEED = 42


def price_momentum_signal(stock_data, target_date, short_half_life=3, long_half_life=14, lookback_days=30):
    # Mirrors calculate_historical_score's decay math, but applied to the stock's own
    # daily returns instead of sentiment positivity, so the mechanism is apples-to-apples
    # even though the scale differs.
    window = stock_data.loc[target_date - timedelta(days=lookback_days): target_date - timedelta(days=1)]
    returns = window['Close'].pct_change().dropna()
    if returns.empty:
        return None

    days_ago = (target_date - returns.index).days.values.astype(float)

    def decayed_ema(half_life):
        weights = 0.5 ** (days_ago / half_life)
        total_weight = weights.sum()
        if total_weight == 0:
            return 0.0
        return float((returns.values * weights).sum() / total_weight)

    return decayed_ema(short_half_life) - decayed_ema(long_half_life)


def build_shuffled_positivities(articles_by_date, symbol_index):
    # Flatten all (date, positivity) pairs collected for this symbol, permute the
    # positivity values with a fixed seed, then redistribute them back into the same
    # per-date article-count structure. Article dates/counts-per-window are untouched;
    # only which score attaches to which slot is scrambled, so trade frequency stays
    # roughly the same going into calculate_historical_score, while any genuine
    # within-window predictive relationship is destroyed.
    rng = np.random.RandomState(SHUFFLE_SEED + symbol_index)
    all_positivities = []
    for _, positivities in articles_by_date.values():
        all_positivities.extend(positivities)
    shuffled = np.array(all_positivities)
    rng.shuffle(shuffled)

    result = {}
    cursor = 0
    for target_date, (articles, positivities) in articles_by_date.items():
        n = len(positivities)
        result[target_date] = shuffled[cursor:cursor + n].tolist()
        cursor += n
    return result


def gather_raw_signals():
    # One pass over (symbol, date): fetch news + price data once, compute sentiment
    # momentum and price momentum together, and stash per-symbol article/positivity
    # data so the shuffle arm can be built without re-fetching or re-scoring anything.
    print("Gathering raw signals for all three arms...")
    model, tokenizer = load_model()
    start_date, end_date, test_dates = backtest.get_test_window()

    per_symbol = {}  # symbol -> {'stock_data':..., 'dates':[], 'sentiment_momentum':[], 'price_momentum':[], 'articles_by_date': {}}

    for symbol in backtest.STOCKS:
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(start=(start_date - timedelta(days=30)).strftime('%Y-%m-%d'), end=pd.Timestamp.now().strftime('%Y-%m-%d'))
        if stock_data.empty:
            continue
        stock_data.index = stock_data.index.tz_localize(None)
        stock_data['ATR'] = calculate_atr(stock_data, period=14)

        dates, sentiment_momenta, price_momenta = [], [], []
        articles_by_date = {}

        for target_date in test_dates:
            to_date = target_date.strftime('%Y-%m-%d')
            from_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            news = fetch_finnhub_news(symbol, from_date, to_date)
            if not news:
                continue

            sampled_news = sample_articles(news, max_count=40)
            sampled_positivity = score_articles(sampled_news, model, tokenizer)
            articles_by_date[target_date] = (sampled_news, sampled_positivity)

            short_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 3)
            long_term = calculate_historical_score(sampled_news, sampled_positivity, target_date, 14)
            sentiment_momentum = short_term - long_term

            price_momentum = price_momentum_signal(stock_data, target_date)
            if price_momentum is None:
                continue

            dates.append(target_date)
            sentiment_momenta.append(sentiment_momentum)
            price_momenta.append(price_momentum)

        per_symbol[symbol] = {
            'stock_data': stock_data,
            'dates': dates,
            'sentiment_momentum': sentiment_momenta,
            'price_momentum': price_momenta,
            'articles_by_date': articles_by_date,
        }
        print(f"  {symbol}: {len(dates)} usable dates")

    return per_symbol, start_date, end_date


def calibrate_threshold(values, target_rate_bullish, target_rate_bearish):
    values = np.array(values)
    bullish_cutoff = np.quantile(values, 1 - target_rate_bullish) if target_rate_bullish > 0 else float('inf')
    bearish_cutoff = np.quantile(values, target_rate_bearish) if target_rate_bearish > 0 else float('-inf')
    return bullish_cutoff, bearish_cutoff


def run_arm(arm_name, per_symbol, momentum_key, bullish_threshold, bearish_threshold, start_date, end_date):
    trade_records = []
    total_trades = 0
    winning_trades = 0

    for symbol, data in per_symbol.items():
        stock_data = data['stock_data']
        for target_date, momentum in zip(data['dates'], data[momentum_key]):
            is_bullish = momentum > bullish_threshold
            is_bearish = momentum < bearish_threshold
            if not is_bullish and not is_bearish:
                continue

            trade = simulate_trade(
                stock_data, target_date, is_bullish, is_bearish,
                backtest.ATR_MULTIPLIER, backtest.TAKE_PROFIT_MULTIPLIER, backtest.TRANSACTION_COST_PCT,
                backtest.RISK_PER_TRADE_PCT, backtest.MAX_POSITION_WEIGHT, backtest.HOLDING_PERIOD_DAYS,
            )
            if trade is None:
                continue

            total_trades += 1
            if trade['trade_return'] > 0:
                winning_trades += 1
            trade_records.append(trade)

    if total_trades == 0:
        return {'arm': arm_name, 'trades': 0, 'win_rate': 0.0, 'metrics': None}

    metrics = compute_performance_metrics(trade_records, start_date, end_date)
    return {
        'arm': arm_name,
        'trades': total_trades,
        'win_rate': winning_trades / total_trades,
        'metrics': metrics,
    }


def run_ablation():
    per_symbol, start_date, end_date = gather_raw_signals()

    # Build the shuffled-sentiment arm's momentum series from arm 1's already-scored
    # positivities, per symbol, with a fixed seed for reproducibility.
    for symbol_index, (symbol, data) in enumerate(per_symbol.items()):
        shuffled_positivities = build_shuffled_positivities(data['articles_by_date'], symbol_index)
        shuffled_momenta = []
        for target_date in data['dates']:
            positivities = shuffled_positivities[target_date]
            articles, _ = data['articles_by_date'][target_date]
            short_term = calculate_historical_score(articles, positivities, target_date, 3)
            long_term = calculate_historical_score(articles, positivities, target_date, 14)
            shuffled_momenta.append(short_term - long_term)
        data['shuffled_sentiment_momentum'] = shuffled_momenta

    all_sentiment = [m for d in per_symbol.values() for m in d['sentiment_momentum']]
    all_price = [m for d in per_symbol.values() for m in d['price_momentum']]
    all_shuffled = [m for d in per_symbol.values() for m in d['shuffled_sentiment_momentum']]

    bullish_rate = float(np.mean(np.array(all_sentiment) > backtest.MOMENTUM_BULLISH_THRESHOLD))
    bearish_rate = float(np.mean(np.array(all_sentiment) < backtest.MOMENTUM_BEARISH_THRESHOLD))
    print(f"\nSentiment arm empirical trade rate: bullish={bullish_rate*100:.1f}% bearish={bearish_rate*100:.1f}%")

    # NOTE: these cutoffs are calibrated against the FULL test-window distribution for
    # each control arm, purely so each arm fires at a comparable rate to the sentiment
    # arm. This is a deliberate, narrow form of look-ahead: it only decides where a
    # control arm's line is drawn, never touches the sentiment arm's fixed +/-0.01
    # threshold or any individual trade's entry/exit logic.
    price_bull_cut, price_bear_cut = calibrate_threshold(all_price, bullish_rate, bearish_rate)
    shuffled_bull_cut, shuffled_bear_cut = calibrate_threshold(all_shuffled, bullish_rate, bearish_rate)

    results = []
    results.append(run_arm('Sentiment Momentum', per_symbol, 'sentiment_momentum',
                            backtest.MOMENTUM_BULLISH_THRESHOLD, backtest.MOMENTUM_BEARISH_THRESHOLD,
                            start_date, end_date))
    results.append(run_arm('Price Momentum (control)', per_symbol, 'price_momentum',
                            price_bull_cut, price_bear_cut, start_date, end_date))
    results.append(run_arm('Shuffled Sentiment (placebo)', per_symbol, 'shuffled_sentiment_momentum',
                            shuffled_bull_cut, shuffled_bear_cut, start_date, end_date))

    print("\n" + "=" * 78)
    print(f"{'ARM':<30} {'TRADES':<8} {'WIN%':<8} {'RETURN':<9} {'SHARPE':<8} {'SORTINO':<9} {'MAXDD'}")
    print("=" * 78)
    for r in results:
        if r['metrics'] is None:
            print(f"{r['arm']:<30} {r['trades']:<8} {'--':<8} {'--':<9} {'--':<8} {'--':<9} {'--'}")
            continue
        m = r['metrics']
        print(f"{r['arm']:<30} {r['trades']:<8} {r['win_rate']*100:<7.1f}% {m['total_return']*100:<8.2f}% "
              f"{m['sharpe']:<8.2f} {m['sortino']:<9.2f} {m['max_drawdown']*100:.2f}%")

    benchmark_dates, benchmark_equity = fetch_benchmark_equity(start_date, end_date)
    if benchmark_equity is not None:
        benchmark_return = benchmark_equity.iloc[-1] - 1.0
        print(f"{'SPY Buy & Hold':<30} {'--':<8} {'--':<8} {benchmark_return*100:<8.2f}% {'--':<8} {'--':<9} {'--'}")
    print("=" * 78)

    curves = []
    styles = [{'linewidth': 2.2}, {'linewidth': 1.6}, {'linewidth': 1.6, 'linestyle': ':'}]
    for r, style in zip(results, styles):
        if r['metrics'] is None:
            continue
        curves.append({'label': r['arm'], 'dates': r['metrics']['dates'], 'equity': r['metrics']['equity_curve'], 'style': style})
    if benchmark_equity is not None:
        curves.append({'label': 'SPY Buy & Hold', 'dates': benchmark_dates, 'equity': benchmark_equity,
                        'style': {'linewidth': 1.5, 'alpha': 0.7, 'linestyle': '--', 'color': 'gray'}})

    plot_equity_curve(curves, output_path='ablation_equity_curve.png')


if __name__ == '__main__':
    run_ablation()
