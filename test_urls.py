import pandas as pd
import requests
from cinematch_app import load_data

df = load_data()
for url in df['poster'].head(10):
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        print(f"{r.status_code} - {url}")
    except Exception as e:
        print(f"Error - {url}")
