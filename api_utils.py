import time
import requests

# Finnhub's free tier throttles rapid-fire requests hard enough that a 5s timeout was
# timing out on nearly every call in a tight loop. Retry with backoff plus a floor
# delay between requests fixes that instead of silently losing most of the sample.
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
MIN_REQUEST_INTERVAL = 0.5

def fetch_json(url, error_label):
    for attempt in range(MAX_RETRIES):
        time.sleep(MIN_REQUEST_INTERVAL)
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            print(f"API error for {error_label}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"Request failed for {error_label} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    return []
