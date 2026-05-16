import pandas as pd
from cinematch_app import collaborative_filtering, df, load_data, user_ratings_db

print("Data loaded")
cf = collaborative_filtering(["Dangal"], "Both", 10)
print(cf)
