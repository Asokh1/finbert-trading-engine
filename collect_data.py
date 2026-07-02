import os
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API = os.getenv('FINNHUB_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

STOCKS = ['ICLN', 'CLEAN', 'TAN', 'ENPH', 'RUN', 'PLUG', 'NEE', 'SEDG']
DB_PATH = 'data/sentiment_data.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY, symbol TEXT, headline TEXT, 
                  summary TEXT, source TEXT, date TEXT)''')
    conn.commit()
    conn.close()

def get_finnhub_news(symbol):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&limit=20&token={FINNHUB_API}'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def get_newsapi_data(query):
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={NEWSAPI_KEY}'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get('articles', [])
    except:
        pass
    return []

def save_to_db(symbol, headline, summary, source):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO news (symbol, headline, summary, source, date) VALUES (?, ?, ?, ?, ?)',
              (symbol, headline, summary, source, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def collect_data():
    init_db()
    print("Collecting renewable energy stock news...\n")
    
    for symbol in STOCKS:
        print(f"Fetching {symbol}...")
        news = get_finnhub_news(symbol)
        for item in news:
            save_to_db(symbol, item.get('headline', ''), item.get('summary', ''), 'Finnhub')
    
    query = 'renewable energy solar wind clean energy'
    articles = get_newsapi_data(query)
    for article in articles:
        save_to_db('GENERAL', article['title'], article.get('description', ''), 'NewsAPI')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = c.execute('SELECT COUNT(*) FROM news').fetchone()[0]
    conn.close()
    
    print(f"\nTotal articles collected: {count}")

if __name__ == '__main__':
    collect_data()