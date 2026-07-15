# Quantitative Sentiment Trading Engine

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Local_Inference-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![License](https://img.shields.io/badge/License-MIT-green)

A high-performance algorithmic trading engine designed to execute sentiment-driven strategies utilizing Natural Language Processing (NLP). This system leverages a custom-tuned FinBERT Transformer model to parse financial news, calculate time-decayed sentiment momentum, and systematically backtest strategies with automated risk controls.

## Technology Stack and Machine Learning Models

This project serves as a comprehensive demonstration of full-stack quantitative engineering, featuring hands-on implementation of the following technologies:

* **Deep Learning and NLP Core:** Utilizes the **Hugging Face Transformers** library and **PyTorch** to run local edge inference. The core model is `ProsusAI/finbert`, a specialized BERT architecture pre-trained extensively on financial corpora (10-K reports, financial news) to accurately contextualize market-specific jargon.
* **Parameter-Efficient Fine-Tuning (PEFT):** Implements **LoRA (Low-Rank Adaptation)** to apply custom-trained weights over the base FinBERT model. This demonstrates the ability to adapt massive language models dynamically at runtime, optimizing for CPU-efficient local execution without cloud dependency.
* **Data Engineering:** Manages live data ingestion and rate-limiting utilizing the **Finnhub API** for unstructured news scraping and `yfinance` for historical market data. Employs **Pandas** for rigorous time-series alignment and vector-based financial calculations.

## Quantitative Strategy and Risk Management

* **Mathematical Sentiment Momentum:** The engine moves beyond binary positive/negative classification by applying a Moving Average Convergence Divergence (MACD) approach to the sentiment scores ($S$):
  
  $$MACD_{sentiment} = EMA_{short}(S) - EMA_{long}(S)$$
  
  This identifies genuine trend reversals and momentum breakouts by measuring the delta between the 7-day and 30-day sentiment averages, isolating acceleration rather than static positivity.
* **Dynamic Volatility Risk Management:** Replaces static stop-loss assumptions with a volatility-adaptive framework built on the 14-day **Average True Range (ATR)**. Each position is bracketed by a stop-loss set at $2\times ATR_{14}$ below entry and a take-profit target at $3\times ATR_{14}$ above it, producing an asymmetric 1.5:1 reward-to-risk profile that automatically widens or tightens with the underlying asset's realized volatility.
* **Volatility-Adjusted Position Sizing:** Capital allocated per trade is scaled inversely to each asset's ATR, so every position risks roughly the same fraction of capital if its stop is hit, regardless of how volatile the underlying instrument is (capped to prevent over-leveraging on unusually low-volatility names).
* **Transaction Cost Modeling:** Applies realistic commission and slippage assumptions to both the entry and exit leg of every trade, so reported PnL reflects tradable, cost-adjusted returns rather than frictionless theoretical performance.

## System Architecture

The pipeline is entirely self-contained, prioritizing latency optimization and security by keeping model inference and trading logic on local hardware.

1. **Ingestion:** Fetches chronological market data and news headlines.
2. **Inference:** Loads LoRA weights into FinBERT to classify sentiment polarity and magnitude.
3. **Signal Generation:** Applies the MACD decay function to output Bullish, Bearish, or Neutral signals based on momentum thresholds.
4. **Execution Simulation:** Parses signals through the risk engine to log entries, exits, and capital fluctuations.

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
To calculate the MACD sentiment momentum for a specific ticker over 7-day and 30-day windows:
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
*Outputs a detailed trade ledger, including entry/exit pricing, position size, individual trade returns, and how each trade closed (stopped out, target hit, or time-based exit), followed by aggregate win rate and cost-adjusted Cumulative PnL:*

```
DATE         SYM    SIGNAL                 PRICE IN   PRICE OUT  SIZE     RETURN
=====================================================================================
2026-01-04   AMZN   BEARISH (SHORT)        $233.06    $241.69    0.27   x  -3.91%  [STOPPED OUT]
2026-03-01   AMZN   BULLISH (BUY)          $208.39    $213.49    0.18   x   2.24%
2025-12-21   TSLA   BEARISH (SHORT)        $488.73    $459.64    0.14   x   5.76%
2025-12-28   TSLA   BEARISH (SHORT)        $459.64    $451.67    0.13   x   1.54%
2026-03-29   AAPL   BULLISH (BUY)          $246.40    $261.59    0.24   x   5.95%  [TARGET HIT]
2026-04-05   AAPL   BULLISH (BUY)          $258.62    $248.01    0.24   x  -4.29%  [STOPPED OUT]
2026-03-22   MSFT   BEARISH (SHORT)        $382.17    $360.88    0.27   x   5.38%  [TARGET HIT]
2026-06-14   MSFT   BULLISH (BUY)          $399.76    $372.84    0.15   x  -6.92%  [STOPPED OUT]
2026-01-04   GOOGL  BULLISH (BUY)          $316.13    $331.43    0.26   x   4.63%
=====================================================================================
Total Trades Taken:  18
Winning Trades:      10
Win Rate:            55.6%
Cumulative PnL:      -0.23%
```

*Note: with a small number of trades per run, this output is meant to demonstrate the risk-management mechanics (ATR-adaptive stops/targets, volatility-scaled position sizing, cost-adjusted PnL) rather than serve as a claim of statistical profitability — the focus of this project is the engineering of the risk engine itself, not a curve-fit backtest result.*
