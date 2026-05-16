import os
import time
import requests
from duckduckgo_search import DDGS
from cinematch_app import load_data

os.makedirs("assets/posters", exist_ok=True)
df = load_data()

ddgs = DDGS()

headers = {"User-Agent": "Mozilla/5.0"}

def download_image(url, filepath):
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

for idx, row in df.iterrows():
    title = row['title']
    year = row['year']
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filepath = f"assets/posters/{safe_title}.jpg"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        continue
        
    print(f"Searching for {title} ({year})...")
    query = f"{title} {year} movie poster imdb"
    
    try:
        results = ddgs.images(query, max_results=3)
        success = False
        for res in results:
            img_url = res['image']
            if download_image(img_url, filepath):
                print(f"  -> Downloaded {title}")
                success = True
                break
        if not success:
            print(f"  -> Failed to download {title}")
    except Exception as e:
        print(f"  -> Error: {e}")
        
    time.sleep(1)

print("Done.")
