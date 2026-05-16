import urllib.request
import urllib.parse
import re
import json
import time

from cinematch_app import load_data

def get_trailer_id(title, year):
    query = f"{title} {year} movie trailer official"
    url = 'https://www.youtube.com/results?search_query=' + urllib.parse.quote(query)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read().decode('utf-8')
        video_ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html)
        if video_ids:
            return f"https://www.youtube.com/watch?v={video_ids[0]}"
    except Exception as e:
        print(f"Error fetching {title}: {e}")
    return ""

def main():
    df = load_data()
    movies = df.to_dict('records')
    
    for m in movies:
        print(f"Fetching trailer for {m['title']}...")
        trailer = get_trailer_id(m['title'], m['year'])
        m['trailer'] = trailer
        time.sleep(0.1) # Be nice to YouTube

    with open('movies_with_trailers.json', 'w') as f:
        json.dump(movies, f, indent=4)
        
    print("Saved to movies_with_trailers.json")

if __name__ == "__main__":
    main()
