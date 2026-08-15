import math
from datetime import timedelta
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from signal_engine import calculate_trade_return

def compute_stop_target(price_in, current_atr, is_bullish, atr_multiplier, take_profit_multiplier):
    if is_bullish:
        stop_loss_price = price_in - (atr_multiplier * current_atr)
        take_profit_price = price_in + (take_profit_multiplier * current_atr)
    else: # shorting - stop sits above entry, target sits below
        stop_loss_price = price_in + (atr_multiplier * current_atr)
        take_profit_price = price_in - (take_profit_multiplier * current_atr)
    return stop_loss_price, take_profit_price

def compute_position_weight(price_in, current_atr, atr_multiplier, risk_per_trade_pct, max_position_weight):
    # wider stop = smaller size, so every trade risks about the same amount
    stop_distance_pct = (atr_multiplier * current_atr) / price_in
    return min(risk_per_trade_pct / stop_distance_pct, max_position_weight)

def check_bar_against_levels(row, is_bullish, stop_loss_price, take_profit_price):
    # stop-loss first, in case a gap day blows through both levels
    if is_bullish and row['Low'] < stop_loss_price:
        return 'stop', stop_loss_price
    elif not is_bullish and row['High'] > stop_loss_price:
        return 'stop', stop_loss_price

    if is_bullish and row['High'] > take_profit_price:
        return 'target', take_profit_price
    elif not is_bullish and row['Low'] < take_profit_price:
        return 'target', take_profit_price

    return None, None

def open_trade(stock_data, target_date, is_bullish, atr_multiplier, take_profit_multiplier,
                risk_per_trade_pct, max_position_weight):
    if stock_data.empty:
        return None

    price_in_idx = stock_data.index.get_indexer([target_date], method='nearest')[0]
    actual_entry_date = stock_data.index[price_in_idx]
    price_in = float(stock_data['Close'].iloc[price_in_idx])

    current_atr = stock_data['ATR'].iloc[price_in_idx]
    if pd.isna(current_atr):
        return None # not enough history yet for a 14-day ATR

    # stop and target are both set once at entry, based on that day's ATR
    stop_loss_price, take_profit_price = compute_stop_target(
        price_in, current_atr, is_bullish, atr_multiplier, take_profit_multiplier)
    position_weight = compute_position_weight(
        price_in, current_atr, atr_multiplier, risk_per_trade_pct, max_position_weight)

    return {
        'entry_date': actual_entry_date,
        'price_in': price_in,
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'position_weight': position_weight,
    }

def simulate_trade(stock_data, target_date, is_bullish, is_bearish, atr_multiplier,
                    take_profit_multiplier, transaction_cost_pct, risk_per_trade_pct,
                    max_position_weight, holding_period_days=7):
    if not is_bullish and not is_bearish:
        return None

    entry = open_trade(stock_data, target_date, is_bullish, atr_multiplier, take_profit_multiplier,
                        risk_per_trade_pct, max_position_weight)
    if entry is None:
        return None

    actual_entry_date = entry['entry_date']
    price_in = entry['price_in']
    stop_loss_price = entry['stop_loss_price']
    take_profit_price = entry['take_profit_price']
    position_weight = entry['position_weight']

    # Get the next `holding_period_days` days of price data
    window_end_date = actual_entry_date + timedelta(days=holding_period_days)
    holding_period_data = stock_data.loc[actual_entry_date + timedelta(days=1): window_end_date]

    trade_return = 0.0
    price_out = price_in
    exit_reason = 'time'
    trade_exit_date = actual_entry_date

    # Check price day by day
    for current_date, row in holding_period_data.iterrows():
        reason, level_price = check_bar_against_levels(row, is_bullish, stop_loss_price, take_profit_price)
        if reason is not None:
            price_out = level_price
            trade_return = calculate_trade_return(price_in, price_out, is_bullish, transaction_cost_pct)
            exit_reason = reason
            trade_exit_date = current_date
            break

    # survived the window without hitting either level, close normally
    if exit_reason == 'time':
        if holding_period_data.empty:
            return None
        price_out = float(holding_period_data.iloc[-1]['Close'])
        trade_return = calculate_trade_return(price_in, price_out, is_bullish, transaction_cost_pct)
        trade_exit_date = holding_period_data.index[-1]

    return {
        'entry_date': actual_entry_date,
        'exit_date': trade_exit_date,
        'price_in': price_in,
        'price_out': price_out,
        'position_weight': position_weight,
        'trade_return': trade_return,
        'weighted_return': trade_return * position_weight,
        'exit_reason': exit_reason,
    }

def compute_performance_metrics(trade_records, start_date, end_date):
    # daily, portfolio-level mark-to-market: trades held over the same days must
    # add up on those days, not queue up one-after-another regardless of overlap.
    # each trade's total weighted_return is spread evenly across the business days
    # it was actually open, then every trade open on a given day contributes to
    # that day's portfolio return.
    calendar = pd.bdate_range(start=start_date, end=end_date)
    daily_returns = pd.Series(0.0, index=calendar)

    for t in trade_records:
        held_days = calendar[(calendar > t['entry_date']) & (calendar <= t['exit_date'])]
        if len(held_days) == 0:
            continue
        daily_returns.loc[held_days] += t['weighted_return'] / len(held_days)

    equity_curve = (1 + daily_returns).cumprod()
    running_peak = equity_curve.cummax()
    max_drawdown = ((equity_curve - running_peak) / running_peak).min()

    mean_return = daily_returns.mean()
    std_return = daily_returns.std(ddof=0)
    TRADING_DAYS_PER_YEAR = 252
    sharpe = (mean_return / std_return) * math.sqrt(TRADING_DAYS_PER_YEAR) if std_return > 0 else 0.0

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std(ddof=0) if len(downside_returns) > 1 else 0.0
    sortino = (mean_return / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0

    return {
        'dates': equity_curve.index,
        'equity_curve': equity_curve.values,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'total_return': equity_curve.iloc[-1] - 1.0,
    }

def fetch_benchmark_equity(start_date, end_date, symbol='SPY'):
    benchmark = yf.Ticker(symbol).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if benchmark.empty:
        return None, None
    benchmark.index = benchmark.index.tz_localize(None)
    equity = benchmark['Close'] / benchmark['Close'].iloc[0]
    return benchmark.index, equity

def plot_equity_curve(curves, output_path='equity_curve.png'):
    # curves: list of {'label': str, 'dates': ..., 'equity': ..., 'style': optional dict of plot kwargs}
    plt.figure(figsize=(10, 6))
    for curve in curves:
        style = {'linewidth': 2}
        style.update(curve.get('style', {}))
        plt.plot(curve['dates'], curve['equity'], label=curve['label'], **style)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Strategy Equity Curve vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Equity curve saved to {output_path}")
