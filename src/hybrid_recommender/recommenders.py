"""User-based and item-based movie recommendation algorithms."""

import math

import pandas as pd

RESULT_COLUMNS = ("movieId", "title", "score")


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series(dtype="int64"),
            "title": pd.Series(dtype="object"),
            "score": pd.Series(dtype="float64"),
        }
    )


def _format_results(scores: pd.Series, movies: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return _empty_results()

    movie_scores = scores.rename("score").rename_axis("movieId").reset_index()
    results = movie_scores.merge(
        movies[["movieId", "title"]], on="movieId", how="inner", validate="many_to_one"
    )
    return results.loc[:, RESULT_COLUMNS].sort_values(
        "score", ascending=False, kind="stable", ignore_index=True
    )


def get_user_based_recommendations(
    user_movie_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    count: int = 5,
    min_similarity: float = 0.65,
    min_common_movie_ratio: float = 0.6,
    min_rating: float = 3.5,
) -> pd.DataFrame:
    """Recommend unrated movies based on Pearson-similar users."""
    if count < 1:
        raise ValueError("count must be at least 1.")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1.")
    if not 0 < min_common_movie_ratio <= 1:
        raise ValueError("min_common_movie_ratio must be greater than 0 and at most 1.")
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"User {user_id} has no ratings in the selected movie matrix.")

    target_ratings = user_movie_matrix.loc[user_id].dropna()
    if target_ratings.empty:
        return _empty_results()

    minimum_overlap = math.ceil(len(target_ratings) * min_common_movie_ratio)
    overlap_counts = (
        user_movie_matrix.loc[:, target_ratings.index].notna().sum(axis="columns")
    )
    candidate_ids = overlap_counts[
        (overlap_counts >= minimum_overlap) & (overlap_counts.index != user_id)
    ].index
    if candidate_ids.empty:
        return _empty_results()

    candidate_ratings = user_movie_matrix.loc[candidate_ids, target_ratings.index]
    similarities = candidate_ratings.T.corrwith(target_ratings).dropna()
    similarities = similarities[similarities >= min_similarity]
    if similarities.empty:
        return _empty_results()

    candidate_ratings = user_movie_matrix.loc[similarities.index]
    weighted_ratings = candidate_ratings.mul(similarities, axis="index")
    weight_totals = (
        candidate_ratings.notna().mul(similarities, axis="index").sum(axis="index")
    )
    scores = weighted_ratings.sum(axis="index", min_count=1).div(weight_totals)
    scores = scores.drop(index=target_ratings.index, errors="ignore")
    scores = scores[scores > min_rating].nlargest(count)
    return _format_results(scores, movies)


def get_item_based_recommendations(
    user_movie_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_id: int,
    count: int = 5,
) -> pd.DataFrame:
    """Recommend movies correlated with the user's latest eligible five-star film."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    user_five_star_ratings = ratings[
        (ratings["userId"] == user_id)
        & (ratings["rating"] == 5.0)
        & ratings["movieId"].isin(user_movie_matrix.columns)
    ]
    if user_five_star_ratings.empty:
        return _empty_results()

    latest_timestamp = user_five_star_ratings["timestamp"].max()
    seed_movie_id = int(
        user_five_star_ratings.loc[
            user_five_star_ratings["timestamp"] == latest_timestamp, "movieId"
        ].min()
    )
    seed_ratings = user_movie_matrix[seed_movie_id]
    similarities = user_movie_matrix.corrwith(seed_ratings).dropna()
    watched_movie_ids = ratings.loc[ratings["userId"] == user_id, "movieId"]
    similarities = similarities.drop(index=watched_movie_ids.to_list(), errors="ignore")
    similarities = similarities[similarities > 0].nlargest(count)
    return _format_results(similarities, movies)
