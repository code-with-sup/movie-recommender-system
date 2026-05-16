from duckduckgo_search import DDGS

ddgs = DDGS()
results = ddgs.text("Dangal movie trailer site:youtube.com", max_results=1)
print(results)
