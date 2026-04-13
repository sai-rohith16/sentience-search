from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import sentience_engine
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware # Moved import to the top

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "cleaned_movies.csv"
VECTORS_PATH = DATA_DIR / "movie_vectors.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)


class SearchResult(BaseModel):
    title: str
    vote_average: float
    runtime: float
    overview: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.movies_df = pd.read_csv(CSV_PATH)
    app.state.search_engine = sentience_engine.SearchEngine(str(VECTORS_PATH))
    app.state.model = SentenceTransformer(MODEL_NAME)
    yield

# --- FIX START ---
app = FastAPI(lifespan=lifespan)

# Add the middleware to the ALREADY CREATED app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- FIX END ---


@app.post("/search", response_model=list[SearchResult])
def search(body: SearchQuery, request: Request) -> list[SearchResult]:
    model: SentenceTransformer = request.app.state.model
    engine: sentience_engine.SearchEngine = request.app.state.search_engine
    movies_df: pd.DataFrame = request.app.state.movies_df

    vec = model.encode(body.query, convert_to_numpy=True)
    query_vec = np.asarray(vec, dtype=np.float32).reshape(-1)

    indices = engine.search(query_vec, 5)

    out: list[SearchResult] = []
    for idx in np.asarray(indices).flat:
        row = movies_df.iloc[int(idx)]
        out.append(
            SearchResult(
                title=str(row["title"]),
                vote_average=float(row["vote_average"]),
                runtime=float(row["runtime"]),
                overview=str(row["overview"]),
            )
        )
    return out