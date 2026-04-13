import pandas as pd

# Load the fresh file you just downloaded
df = pd.read_csv('data/tmdb_5000_movies.csv')

# Select the 5 columns we need
cleaned_df = df[['id', 'title', 'overview', 'vote_average', 'runtime']].dropna()

# Save it
cleaned_df.to_csv('data/cleaned_movies.csv', index=False)
print("✅ Success! Fresh cleaned_movies.csv created.")