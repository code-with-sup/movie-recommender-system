import re
import requests
import time

with open("cinematch_app.py", "r") as f:
    content = f.read()

def get_omdb_poster(title, year):
    # Some bollywood titles might need exact search
    url = f"http://www.omdbapi.com/?t={title}&y={year}&apikey=thewdb"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data["Poster"]
    except:
        pass
    # Try without year
    url = f"http://www.omdbapi.com/?t={title}&apikey=thewdb"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data["Poster"]
    except:
        pass
    return None

new_content = content
matches = re.finditer(r'\{"title":"([^"]+)".*?"year":(\d+).*?"poster":"([^"]+)"\}', content)

count = 0
for m in matches:
    full_str = m.group(0)
    title = m.group(1)
    year = m.group(2)
    old_poster = m.group(3)
    
    poster = get_omdb_poster(title, year)
    if poster:
        new_str = full_str.replace(old_poster, poster)
        new_content = new_content.replace(full_str, new_str)
        print(f"Updated: {title}")
    else:
        print(f"Failed to find OMDB poster for: {title}")
    time.sleep(0.1)
    count += 1

with open("cinematch_app.py", "w") as f:
    f.write(new_content)

print(f"Done. Processed {count} movies.")
