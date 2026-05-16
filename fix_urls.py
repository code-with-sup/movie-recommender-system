import wikipedia
import urllib.request
from cinematch_app import load_data
import time

df = load_data()
updated_urls = {}

for idx, row in df.iterrows():
    url = row['poster']
    title = row['title']
    year = row['year']
    
    # check if url is broken
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        if "404" in str(e):
            print(f"{title} is broken (404), trying to fix...")
            try:
                page = wikipedia.page(f"{title} ({year} film)")
                images = [i for i in page.images if i.lower().endswith(('.jpg', '.png', '.jpeg')) and 'poster' in i.lower()]
                if not images:
                    images = [i for i in page.images if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    new_url = images[0]
                    updated_urls[title] = new_url
                    print(f"Fixed {title} -> {new_url}")
                else:
                    print(f"No image found for {title}")
            except Exception as e2:
                print(f"Error fetching wiki for {title}: {e2}")
            time.sleep(1)

print("Updated URLs:")
for t, u in updated_urls.items():
    print(f'"{t}": "{u}",')
