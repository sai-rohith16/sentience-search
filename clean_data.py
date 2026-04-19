import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path("data/tmdb_5000_movies.csv")
OUTPUT_PATH = Path("data/cleaned_movies.csv")

# Logic
df = pd.read_csv(INPUT_PATH)
cleaned_df = df[['title', 'vote_average', 'runtime', 'overview']].dropna()
cleaned_df.to_csv(OUTPUT_PATH, index=False)

print(f"Success! Created {OUTPUT_PATH} with {len(cleaned_df)} movies.")