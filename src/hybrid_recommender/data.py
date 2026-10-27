"""MovieLens data loading and user-movie matrix construction."""

from pathlib import Path

import pandas as pd

MOVIE_COLUMNS = ("movieId", "title")
RATING_COLUMNS = ("userId", "movieId", "rating", "timestamp")


def _require_columns(
    dataframe: pd.DataFrame, required: tuple[str, ...], source: str
) -> None:
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_movielens_data(
    movies_path: str | Path, ratings_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate MovieLens movie and rating CSV files."""
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    _require_columns(movies, MOVIE_COLUMNS, "Movies file")
    _require_columns(ratings, RATING_COLUMNS, "Ratings file")

    if movies["movieId"].duplicated().any():
        raise ValueError("Movies file contains duplicate movieId values.")
    if movies[["movieId", "title"]].isna().any().any():
        raise ValueError("Movies file contains missing movieId or title values.")
    if ratings[list(RATING_COLUMNS)].isna().any().any():
        raise ValueError("Ratings file contains missing required values.")

    return movies, ratings


def create_user_movie_matrix(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_movie_ratings: int = 1000,
) -> pd.DataFrame:
    """Return a userId-by-movieId matrix for sufficiently rated known movies."""
    if min_movie_ratings < 1:
        raise ValueError("min_movie_ratings must be at least 1.")
    _require_columns(movies, MOVIE_COLUMNS, "Movies data")
    _require_columns(ratings, ("userId", "movieId", "rating"), "Ratings data")

    if movies["movieId"].duplicated().any():
        raise ValueError("Movies data contains duplicate movieId values.")

    popular_movie_ids = ratings["movieId"].value_counts()
    popular_movie_ids = popular_movie_ids[popular_movie_ids >= min_movie_ratings].index
    known_movie_ids = movies.loc[movies["movieId"].isin(popular_movie_ids), "movieId"]
    filtered_ratings = ratings[ratings["movieId"].isin(known_movie_ids)]

    return filtered_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    ).sort_index()
