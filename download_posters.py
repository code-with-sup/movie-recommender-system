import os
import requests
import time
from cinematch_app import load_data

os.makedirs('assets/posters', exist_ok=True)
df = load_data()

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

for idx, row in df.iterrows():
    title = row['title']
    url = row['poster']
    # sanitize filename
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filepath = f"assets/posters/{safe_title}.jpg"
    
    if not os.path.exists(filepath):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded: {title}")
            else:
                print(f"Failed ({r.status_code}): {title}")
            time.sleep(0.1) # Be nice
        except Exception as e:
            print(f"Error downloading {title}: {e}")
    else:
        print(f"Already exists: {title}")

print("Done downloading posters.")
