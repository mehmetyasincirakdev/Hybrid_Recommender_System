import pandas as pd
import pytest

from hybrid_recommender.recommenders import (
    get_item_based_recommendations,
    get_user_based_recommendations,
)


@pytest.fixture
def recommendation_data():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["One", "Two", "Three", "Four"],
        }
    )
    matrix = pd.DataFrame(
        {
            1: [5.0, 5.0, 4.0],
            2: [4.0, 4.0, 3.0],
            3: [5.0, 4.0, 4.0],
            4: [float("nan"), 5.0, 4.0],
        },
        index=pd.Index([10, 20, 30], name="userId"),
    )
    ratings = pd.DataFrame(
        {
            "userId": [10, 10],
            "movieId": [1, 3],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        }
    )
    return movies, matrix, ratings


def test_user_based_recommendations_exclude_watched_movies(recommendation_data):
    movies, matrix, _ = recommendation_data

    results = get_user_based_recommendations(
        matrix,
        movies,
        user_id=10,
        min_similarity=0.5,
        min_common_movie_ratio=0.6,
        min_rating=3.0,
    )

    assert results["movieId"].tolist() == [4]
    assert results["title"].tolist() == ["Four"]


def test_user_based_recommendations_reject_unknown_user(recommendation_data):
    movies, matrix, _ = recommendation_data

    with pytest.raises(ValueError, match="User 999"):
        get_user_based_recommendations(matrix, movies, user_id=999)


def test_item_based_recommendations_use_latest_five_star_movie(recommendation_data):
    movies, matrix, ratings = recommendation_data

    results = get_item_based_recommendations(matrix, movies, ratings, user_id=10)

    assert results["movieId"].tolist() == [2]
    assert 1 not in results["movieId"].tolist()
    assert 3 not in results["movieId"].tolist()
    assert results.iloc[0]["title"] == "Two"


def test_item_based_recommendations_return_empty_without_seed(recommendation_data):
    movies, matrix, ratings = recommendation_data

    results = get_item_based_recommendations(matrix, movies, ratings, user_id=999)

    assert results.empty
    assert results.columns.tolist() == ["movieId", "title", "score"]
