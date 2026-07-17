# Quantitative Sentiment Trading Engine

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Local_Inference-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![License](https://img.shields.io/badge/License-MIT-green)

A high-performance algorithmic trading engine designed to execute sentiment-driven strategies utilizing Natural Language Processing (NLP). This system leverages a custom-tuned FinBERT Transformer model to parse financial news, calculate time-decayed sentiment momentum, and systematically backtest strategies with automated risk controls.

## Technology Stack and Machine Learning Models

This project serves as a comprehensive demonstration of full-stack quantitative engineering, featuring hands-on implementation of the following technologies:

* **Deep Learning and NLP Core:** Utilizes the **Hugging Face Transformers** library and **PyTorch** to run local edge inference. The core model is `ProsusAI/finbert`, a specialized BERT architecture pre-trained extensively on financial corpora (10-K reports, financial news) to accurately contextualize market-specific jargon.
* **Parameter-Efficient Fine-Tuning (PEFT):** Implements **LoRA (Low-Rank Adaptation)** to apply custom-trained weights over the base FinBERT model. Fine-tuning targets FinBERT's native positive/negative/neutral classification head directly, rather than discarding it in favor of a randomly-initialized binary head, so the ~0.3% of parameters LoRA updates build on top of FinBERT's existing financial-sentiment pretraining instead of replacing it. This demonstrates the ability to adapt massive language models dynamically at runtime, optimizing for CPU-efficient local execution without cloud dependency.
* **Data Engineering:** Manages live data ingestion and rate-limiting utilizing the **Finnhub API** for unstructured news scraping and `yfinance` for historical market data. Employs **Pandas** for rigorous time-series alignment and vector-based financial calculations.

## Quantitative Strategy and Risk Management

* **Mathematical Sentiment Momentum:** Each headline is classified positive/negative/neutral, then combined into a single positivity score that blends neutral probability mass toward a 0.5 baseline so an unopinionated headline isn't miscounted as a directional signal. The engine then applies a Moving Average Convergence Divergence (MACD) approach to this sentiment score ($S$):
  
  $$MACD_{sentiment} = EMA_{short}(S) - EMA_{long}(S)$$
  
  This identifies genuine trend reversals and momentum breakouts by measuring the delta between 3-day and 14-day half-life EWMA sentiment averages, isolating acceleration rather than static positivity.
* **Dynamic Volatility Risk Management:** Replaces static stop-loss assumptions with a volatility-adaptive framework built on the 14-day **Average True Range (ATR)**. Each position is bracketed by a stop-loss set at $2\times ATR_{14}$ below entry and a take-profit target at $3\times ATR_{14}$ above it, producing an asymmetric 1.5:1 reward-to-risk profile that automatically widens or tightens with the underlying asset's realized volatility.
* **Volatility-Adjusted Position Sizing:** Capital allocated per trade is scaled inversely to each asset's ATR, so every position risks roughly the same fraction of capital if its stop is hit, regardless of how volatile the underlying instrument is (capped to prevent over-leveraging on unusually low-volatility names).
* **Transaction Cost Modeling:** Applies realistic commission and slippage assumptions to both the entry and exit leg of every trade, so reported PnL reflects tradable, cost-adjusted returns rather than frictionless theoretical performance.
* **Risk-Adjusted Performance Evaluation:** Beyond raw PnL, the backtester marks the portfolio to market on every business day: each trade's return is spread across the business days it was actually held, and multiple trades open on the same day (across different stocks, or opposing signals on the same stock) are summed rather than queued up sequentially as if only one position could ever be open at a time. This daily series is compounded into an equity curve and used to report **Sharpe Ratio**, **Sortino Ratio** (downside-only volatility), and **Maximum Drawdown**, annualized off the standard 252 trading-day convention rather than raw trade count. Results are benchmarked against a **SPY buy-and-hold** equity curve over the identical period to report alpha, rather than presenting the strategy's return in isolation.

## System Architecture

The pipeline is entirely self-contained, prioritizing latency optimization and security by keeping model inference and trading logic on local hardware.

1. **Ingestion:** Fetches chronological market data and news headlines.
2. **Inference:** Loads LoRA weights into FinBERT to classify sentiment polarity and magnitude.
3. **Signal Generation:** Applies the MACD decay function to output Bullish, Bearish, or Neutral signals based on momentum thresholds.
4. **Execution Simulation:** Parses signals through the risk engine to log entries, exits, and capital fluctuations.
5. **Performance Evaluation:** Aggregates trade-level returns into a compounded equity curve, computes Sharpe/Sortino/Max Drawdown, and plots the result against a SPY buy-and-hold benchmark.

## Installation and Setup

**1. Clone the repository**
```bash
git clone https://github.com/Asokh1/finbert-trading-engine.git
cd finbert-trading-engine
```

**2. Initialize the Virtual Environment**
```bash
python -m venv .venv
```
*Windows Authorization and Activation:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```
*macOS/Linux Activation:*
```bash
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Environment Configuration**
Create a `.env` file in the root directory and configure your Finnhub API key:
```env
FINNHUB_API_KEY=your_api_key_here
```

## Usage Documentation

### Momentum Analysis
To calculate the MACD sentiment momentum for a specific ticker over 3-day and 14-day half-life EWMA windows:
```bash
python momentum.py
```
*Outputs the short-term and long-term averages alongside the MACD momentum value and a calculated trend signal.*

### Live Sentiment Analysis
To analyze the current instantaneous sentiment for a specific equity:
```bash
python live_sentiment.py
```
*Outputs the raw negative/positive classification and probability scores based on the most recent news articles.*

### Strategy Backtesting
To run the historical simulation across a portfolio of assets, applying the ATR-based stop-loss/take-profit and volatility-adjusted position sizing:
```bash
python backtest.py
```
*Outputs a detailed trade ledger, including entry/exit pricing, position size, individual trade returns, and how each trade closed (stopped out, target hit, or time-based exit), followed by aggregate win rate, cost-adjusted Cumulative PnL, and risk-adjusted performance metrics benchmarked against SPY buy-and-hold:*

```
DATE         SYM    SIGNAL                 PRICE IN   PRICE OUT  SIZE     RETURN
=====================================================================================
2025-09-14   AMZN   BULLISH (BUY)          $231.43    $227.63    0.24   x  -1.84%
2025-10-12   AMZN   BEARISH (SHORT)        $220.07    $216.48    0.22   x   1.43%
2025-10-26   AMZN   BEARISH (SHORT)        $226.97    $238.18    0.20   x  -5.15%  [STOPPED OUT]
2025-11-02   AMZN   BULLISH (BUY)          $254.00    $239.38    0.17   x  -5.95%  [STOPPED OUT]
2025-11-09   AMZN   BEARISH (SHORT)        $248.40    $232.87    0.15   x   6.06%
2025-11-30   AMZN   BULLISH (BUY)          $233.88    $226.89    0.19   x  -3.18%
2025-12-14   AMZN   BULLISH (BUY)          $222.54    $228.43    0.25   x   2.44%
2026-03-15   AMZN   BULLISH (BUY)          $211.74    $210.14    0.19   x  -0.95%
2026-06-28   AMZN   BEARISH (SHORT)        $240.14    $244.16    0.13   x  -1.88%
2026-07-05   AMZN   BULLISH (BUY)          $244.16    $247.31    0.14   x   1.09%
2025-08-03   TSLA   BEARISH (SHORT)        $309.26    $335.88    0.12   x  -8.83%  [STOPPED OUT]
2025-08-10   TSLA   BULLISH (BUY)          $339.03    $335.16    0.13   x  -1.34%
2025-08-31   TSLA   BEARISH (SHORT)        $329.36    $352.73    0.14   x  -7.31%  [STOPPED OUT]
2025-09-07   TSLA   BULLISH (BUY)          $346.40    $384.32    0.14   x  10.72%  [TARGET HIT]
2025-09-14   TSLA   BULLISH (BUY)          $410.04    $434.21    0.14   x   5.68%
2025-10-12   TSLA   BULLISH (BUY)          $435.90    $447.43    0.10   x   2.44%
2025-10-26   TSLA   BEARISH (SHORT)        $452.42    $468.37    0.12   x  -3.73%
2025-11-02   TSLA   BEARISH (SHORT)        $468.37    $445.23    0.13   x   4.75%
2025-12-14   TSLA   BULLISH (BUY)          $475.31    $488.73    0.17   x   2.62%
2025-12-21   TSLA   BEARISH (SHORT)        $488.73    $459.64    0.14   x   5.76%
2026-01-04   TSLA   BEARISH (SHORT)        $451.67    $448.96    0.13   x   0.40%
2026-05-17   TSLA   BEARISH (SHORT)        $409.99    $426.01    0.12   x  -4.12%
2025-08-10   AAPL   BULLISH (BUY)          $226.54    $230.24    0.20   x   1.43%
2025-09-14   AAPL   BEARISH (SHORT)        $236.03    $245.27    0.26   x  -4.12%  [STOPPED OUT]
2025-10-05   AAPL   BEARISH (SHORT)        $255.97    $246.96    0.27   x   3.32%
2025-10-19   AAPL   BULLISH (BUY)          $261.50    $268.05    0.25   x   2.30%
2025-11-02   AAPL   BULLISH (BUY)          $268.29    $268.93    0.25   x   0.04%
2025-11-09   AAPL   BEARISH (SHORT)        $268.93    $266.96    0.26   x   0.53%
2025-11-23   AAPL   BULLISH (BUY)          $275.41    $282.58    0.23   x   2.40%
2025-12-07   AAPL   BULLISH (BUY)          $277.37    $273.60    0.25   x  -1.56%
2025-12-21   AAPL   BULLISH (BUY)          $270.47    $273.25    0.30   x   0.83%
2025-12-28   AAPL   BULLISH (BUY)          $273.25    $266.76    0.34   x  -2.57%
2026-01-04   AAPL   BEARISH (SHORT)        $266.76    $259.77    0.32   x   2.43%
2026-02-15   AAPL   BULLISH (BUY)          $263.64    $271.89    0.18   x   2.92%
2026-03-22   AAPL   BEARISH (SHORT)        $251.26    $246.40    0.24   x   1.74%
2026-04-19   AAPL   BEARISH (SHORT)        $272.80    $267.36    0.22   x   1.80%
2026-06-14   AAPL   BULLISH (BUY)          $296.42    $297.01    0.19   x  -0.00%
2025-08-17   MSFT   BULLISH (BUY)          $513.00    $501.09    0.22   x  -2.52%
2025-09-07   MSFT   BEARISH (SHORT)        $495.06    $509.71    0.34   x  -3.16%  [STOPPED OUT]
2025-10-19   MSFT   BEARISH (SHORT)        $513.54    $530.09    0.31   x  -3.43%  [STOPPED OUT]
2025-11-09   MSFT   BEARISH (SHORT)        $502.82    $504.30    0.24   x  -0.50%
2025-11-16   MSFT   BEARISH (SHORT)        $504.30    $471.95    0.23   x   6.23%  [TARGET HIT]
2026-01-11   MSFT   BEARISH (SHORT)        $475.06    $455.62    0.37   x   3.90%  [TARGET HIT]
2026-04-05   MSFT   BULLISH (BUY)          $372.07    $383.54    0.23   x   2.88%
2025-08-10   GOOGL  BULLISH (BUY)          $200.43    $202.92    0.24   x   1.04%
2025-11-09   GOOGL  BULLISH (BUY)          $289.53    $272.66    0.17   x  -6.02%  [STOPPED OUT]
2025-11-30   GOOGL  BEARISH (SHORT)        $314.28    $313.31    0.13   x   0.11%
2026-01-04   GOOGL  BULLISH (BUY)          $316.13    $331.43    0.26   x   4.63%
2026-01-18   GOOGL  BULLISH (BUY)          $321.58    $334.12    0.21   x   3.69%
=====================================================================================
Total Trades Taken:  49
Winning Trades:      29
Win Rate:            59.2%
Cumulative PnL:      4.20%
Sharpe Ratio:        1.79
Sortino Ratio:       2.83
Max Drawdown:        -3.67%
Benchmark (SPY B&H): 20.95%
Alpha vs Benchmark:  -16.74%
Equity curve saved to equity_curve.png
```

![Strategy equity curve vs SPY buy-and-hold benchmark](equity_curve.png)

*Note: the backtest window is set to ~355 days — close to the maximum history Finnhub's free-tier company-news endpoint actually serves (empirically confirmed: requests further back than ~365 days return empty results). Widening the window from an earlier ~200-day version roughly tripled the trade count (from 18 to 49), giving a meaningfully more stable read on win rate and risk-adjusted metrics than a smaller sample allows. The signal threshold (`|momentum| > 0.01`) was set from this same window's empirical momentum distribution rather than a strictly disjoint train/validation split; a true holdout test (tuning on the first half of the window, evaluating on the second) showed the momentum score's typical magnitude is not stationary over time, so a fixed threshold tuned on one period can under- or over-fire on another — the same reason the risk engine uses a rolling ATR rather than a fixed-dollar stop, a refinement not yet applied to the sentiment threshold itself. Over this window the strategy underperformed SPY buy-and-hold (negative alpha), largely because SPY had an unusually strong ~21% run and a selective long/short strategy taking 49 total positions across 5 stocks was never structurally positioned to capture that much broad-market beta. The focus of this project is the engineering of the risk and evaluation pipeline itself (ATR-adaptive stops/targets, volatility-scaled position sizing, cost-adjusted PnL, daily portfolio-level mark-to-market accounting for overlapping positions, and honest benchmark-relative reporting), not a curve-fit claim of profitability.*
