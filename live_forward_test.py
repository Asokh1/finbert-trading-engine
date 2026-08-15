import sqlite3
from datetime import datetime, timedelta
import yfinance as yf

from signal_engine import (
    load_model, score_articles, sample_articles, fetch_finnhub_news,
    calculate_historical_score, calculate_atr, calculate_trade_return,
)
from trade_engine import open_trade, check_bar_against_levels, compute_performance_metrics
import backtest

DB_PATH = 'data/forward_test.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    return conn

def init_db(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        signal TEXT NOT NULL,
        momentum REAL NOT NULL,
        predicted_at TEXT NOT NULL,
        entry_date TEXT NOT NULL,
        price_in REAL NOT NULL,
        stop_loss_price REAL NOT NULL,
        take_profit_price REAL NOT NULL,
        position_weight REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        exit_date TEXT,
        price_out REAL,
        exit_reason TEXT,
        trade_return REAL,
        weighted_return REAL,
        closed_at TEXT
    )''')
    conn.execute('''CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_per_symbol
        ON predictions(symbol) WHERE status = 'open' ''')
    conn.commit()

def has_open_position(conn, symbol):
    row = conn.execute("SELECT 1 FROM predictions WHERE symbol = ? AND status = 'open' LIMIT 1", (symbol,)).fetchone()
    return row is not None

def fetch_price_history(symbol, lookback_days=60):
    ticker = yf.Ticker(symbol)
    start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    stock_data = ticker.history(start=start, end=datetime.now().strftime('%Y-%m-%d'))
    if stock_data.empty:
        return stock_data
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data

def record_new_predictions(conn, model, tokenizer):
    print("Checking for new signals...")
    now = datetime.now()

    for symbol in backtest.STOCKS:
        if has_open_position(conn, symbol):
            print(f"{symbol}: already has an open prediction, skipping")
            continue

        try:
            to_date = now.strftime('%Y-%m-%d')
            from_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            news = fetch_finnhub_news(symbol, from_date, to_date)
            if not news:
                print(f"{symbol}: no news, skipping")
                continue

            sampled_news = sample_articles(news, max_count=40)
            sampled_positivity = score_articles(sampled_news, model, tokenizer)

            short_term = calculate_historical_score(sampled_news, sampled_positivity, now, 3)
            long_term = calculate_historical_score(sampled_news, sampled_positivity, now, 14)
            momentum = short_term - long_term

            is_bullish = momentum > backtest.MOMENTUM_BULLISH_THRESHOLD
            is_bearish = momentum < backtest.MOMENTUM_BEARISH_THRESHOLD
            if not is_bullish and not is_bearish:
                print(f"{symbol}: no signal (momentum={momentum:.4f})")
                continue

            stock_data = fetch_price_history(symbol)
            if stock_data.empty:
                print(f"{symbol}: no price data, skipping")
                continue
            stock_data['ATR'] = calculate_atr(stock_data, period=14)

            entry = open_trade(
                stock_data, now, is_bullish,
                backtest.ATR_MULTIPLIER, backtest.TAKE_PROFIT_MULTIPLIER,
                backtest.RISK_PER_TRADE_PCT, backtest.MAX_POSITION_WEIGHT,
            )
            if entry is None:
                print(f"{symbol}: ATR not warmed up yet, skipping")
                continue

            signal_label = 'BULLISH' if is_bullish else 'BEARISH'
            conn.execute('''INSERT INTO predictions
                (symbol, signal, momentum, predicted_at, entry_date, price_in,
                 stop_loss_price, take_profit_price, position_weight, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')''', (
                symbol, signal_label, momentum, now.isoformat(),
                entry['entry_date'].strftime('%Y-%m-%d'), entry['price_in'],
                entry['stop_loss_price'], entry['take_profit_price'], entry['position_weight'],
            ))
            conn.commit()
            print(f"{symbol}: recorded new {signal_label} prediction @ ${entry['price_in']:.2f} "
                  f"(stop ${entry['stop_loss_price']:.2f} / target ${entry['take_profit_price']:.2f})")

        except Exception as e:
            print(f"{symbol}: failed to evaluate signal: {e}")

def check_and_close_open_positions(conn):
    print("\nChecking open predictions...")
    open_rows = conn.execute("SELECT * FROM predictions WHERE status = 'open'").fetchall()
    columns = [d[0] for d in conn.execute("SELECT * FROM predictions LIMIT 0").description]

    now = datetime.now()

    for raw_row in open_rows:
        row = dict(zip(columns, raw_row))
        symbol = row['symbol']
        is_bullish = row['signal'] == 'BULLISH'
        entry_date = datetime.strptime(row['entry_date'], '%Y-%m-%d')

        try:
            stock_data = fetch_price_history(symbol, lookback_days=max(60, (now - entry_date).days + 10))
            if stock_data.empty:
                continue

            bars = stock_data.loc[entry_date + timedelta(days=1): now]

            exit_reason = None
            price_out = None
            trade_exit_date = None

            for current_date, bar in bars.iterrows():
                reason, level_price = check_bar_against_levels(bar, is_bullish, row['stop_loss_price'], row['take_profit_price'])
                if reason is not None:
                    exit_reason = reason
                    price_out = level_price
                    trade_exit_date = current_date
                    break

            if exit_reason is None and (now - entry_date).days > backtest.HOLDING_PERIOD_DAYS and not bars.empty:
                exit_reason = 'time'
                price_out = float(bars.iloc[-1]['Close'])
                trade_exit_date = bars.index[-1]

            if exit_reason is None:
                print(f"{symbol}: still open (entered {row['entry_date']})")
                continue

            trade_return = calculate_trade_return(row['price_in'], price_out, is_bullish, backtest.TRANSACTION_COST_PCT)
            weighted_return = trade_return * row['position_weight']

            conn.execute('''UPDATE predictions SET status='closed', exit_date=?, price_out=?,
                exit_reason=?, trade_return=?, weighted_return=?, closed_at=? WHERE id=?''', (
                trade_exit_date.strftime('%Y-%m-%d'), price_out, exit_reason,
                trade_return, weighted_return, now.isoformat(), row['id'],
            ))
            conn.commit()
            print(f"{symbol}: closed [{exit_reason}] @ ${price_out:.2f} -> {trade_return*100:.2f}%")

        except Exception as e:
            print(f"{symbol}: failed to check open position: {e}")

def report_performance(conn):
    print("\n" + "=" * 60)
    print("FORWARD-TEST TRACK RECORD")
    print("=" * 60)

    closed_rows = conn.execute("SELECT * FROM predictions WHERE status = 'closed'").fetchall()
    columns = [d[0] for d in conn.execute("SELECT * FROM predictions LIMIT 0").description]
    open_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE status = 'open'").fetchone()[0]

    if not closed_rows:
        print(f"No closed predictions yet ({open_count} still open). Check back after some have resolved.")
        return

    trade_records = []
    wins = 0
    for raw_row in closed_rows:
        row = dict(zip(columns, raw_row))
        trade_records.append({
            'entry_date': datetime.strptime(row['entry_date'], '%Y-%m-%d'),
            'exit_date': datetime.strptime(row['exit_date'], '%Y-%m-%d'),
            'weighted_return': row['weighted_return'],
        })
        if row['trade_return'] > 0:
            wins += 1

    start_date = min(t['entry_date'] for t in trade_records)
    end_date = max(t['exit_date'] for t in trade_records)
    metrics = compute_performance_metrics(trade_records, start_date, end_date)

    print(f"Closed Predictions:  {len(closed_rows)}  (still open: {open_count})")
    print(f"Win Rate:            {(wins/len(closed_rows))*100:.1f}%")
    print(f"Cumulative Return:   {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe']:.2f}")
    print(f"Sortino Ratio:       {metrics['sortino']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")

if __name__ == '__main__':
    conn = get_connection()
    model, tokenizer = load_model()
    record_new_predictions(conn, model, tokenizer)
    check_and_close_open_positions(conn)
    report_performance(conn)
    conn.close()
