"""Command-line interface for movie recommendations."""

import argparse
from pathlib import Path

from hybrid_recommender.data import create_user_movie_matrix, load_movielens_data
from hybrid_recommender.recommenders import (
    get_item_based_recommendations,
    get_user_based_recommendations,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate user-based and item-based movie recommendations."
    )
    parser.add_argument("--user-id", type=int, required=True, help="MovieLens user ID.")
    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/raw/movies.csv"),
        help="Path to movies.csv (default: data/raw/movies.csv).",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/raw/ratings.csv"),
        help="Path to ratings.csv (default: data/raw/ratings.csv).",
    )
    parser.add_argument(
        "--min-movie-ratings",
        type=int,
        default=1000,
        help="Minimum rating count required for a movie (default: 1000).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Maximum recommendations per method (default: 5).",
    )
    return parser


def main() -> None:
    """Load data and print recommendations for a MovieLens user."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1.")
    if args.min_movie_ratings < 1:
        parser.error("--min-movie-ratings must be at least 1.")

    try:
        movies, ratings = load_movielens_data(args.movies, args.ratings)
        matrix = create_user_movie_matrix(
            movies, ratings, min_movie_ratings=args.min_movie_ratings
        )
        user_recommendations = get_user_based_recommendations(
            matrix, movies, args.user_id, count=args.count
        )
        item_recommendations = get_item_based_recommendations(
            matrix, movies, ratings, args.user_id, count=args.count
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for label, recommendations in (
        ("User-based recommendations", user_recommendations),
        ("Item-based recommendations", item_recommendations),
    ):
        print(f"\n{label}:")
        if recommendations.empty:
            print("No recommendations available for this user and dataset.")
        else:
            print(recommendations.to_string(index=False))
