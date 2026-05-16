import json

with open('movies_with_trailers.json') as f:
    movies = json.load(f)

with open('cinematch_app.py', 'r') as f:
    content = f.read()

# 1. Replace the movies list
start_idx = content.find('    movies = [')
end_idx = content.find('    ]\n    return pd.DataFrame(movies)', start_idx)

if start_idx != -1 and end_idx != -1:
    new_movies_str = "    movies = [\n"
    for m in movies:
        # Keep quotes consistent
        m_str = json.dumps(m, ensure_ascii=False)
        new_movies_str += f"        {m_str},\n"
    new_movies_str += "    ]"
    
    content = content[:start_idx] + new_movies_str + content[end_idx+5:]

with open('cinematch_app.py', 'w') as f:
    f.write(content)
print("Updated movies list in cinematch_app.py")
