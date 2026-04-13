"""Encode movie overviews with SentenceTransformer and save the embedding matrix as NumPy."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INPUT_PATH = DATA / "cleaned_movies.csv"
OUTPUT_PATH = DATA / "movie_vectors.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    overviews = df["overview"].astype(str).tolist()
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        overviews,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, vectors)


if __name__ == "__main__":
    main()
