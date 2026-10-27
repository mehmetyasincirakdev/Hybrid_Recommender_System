import pandas as pd
import pytest

from hybrid_recommender.data import create_user_movie_matrix, load_movielens_data


def test_create_user_movie_matrix_filters_unpopular_and_unknown_movies() -> None:
    movies = pd.DataFrame(
        {"movieId": [1, 2], "title": ["First", "Second"], "genres": ["Drama", "Comedy"]}
    )
    ratings = pd.DataFrame(
        {
            "userId": [10, 11, 12, 10, 12, 13],
            "movieId": [1, 1, 1, 2, 2, 999],
            "rating": [5.0, 4.0, 3.0, 3.0, 4.0, 2.0],
        }
    )

    matrix = create_user_movie_matrix(movies, ratings, min_movie_ratings=3)

    assert matrix.index.tolist() == [10, 11, 12]
    assert matrix.columns.tolist() == [1]
    assert matrix.loc[10, 1] == 5.0


def test_create_user_movie_matrix_rejects_invalid_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        create_user_movie_matrix(pd.DataFrame(), pd.DataFrame(), min_movie_ratings=0)


def test_load_movielens_data_reports_missing_columns(tmp_path) -> None:
    movies_path = tmp_path / "movies.csv"
    ratings_path = tmp_path / "ratings.csv"
    pd.DataFrame({"movieId": [1]}).to_csv(movies_path, index=False)
    pd.DataFrame(
        {
            "userId": [10],
            "movieId": [1],
            "rating": [5.0],
            "timestamp": [100],
        }
    ).to_csv(ratings_path, index=False)

    with pytest.raises(ValueError, match="title"):
        load_movielens_data(movies_path, ratings_path)
