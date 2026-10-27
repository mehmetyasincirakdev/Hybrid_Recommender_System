"""Movie recommendation algorithms."""

from hybrid_recommender.data import create_user_movie_matrix, load_movielens_data
from hybrid_recommender.recommenders import (
    get_item_based_recommendations,
    get_user_based_recommendations,
)

__all__ = [
    "create_user_movie_matrix",
    "get_item_based_recommendations",
    "get_user_based_recommendations",
    "load_movielens_data",
]
