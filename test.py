import sentience_engine
import numpy as np


vectors = np.load('data/movie_vectors.npy').astype(np.float32)
engine = sentience_engine.SearchEngine(vectors)


results = engine.search(vectors[0], 5) 
print(f"Top 5 movie indices: {results}")