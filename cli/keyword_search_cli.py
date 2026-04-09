import argparse
import json
import string
from tokenize import String


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movie_hits = []
            punct_map = str.maketrans("", "", string.punctuation)

            print(f"Searching for: {args.query}")
            with open("data/movies.json", "r") as f:
                movie_db = json.load(f)
            for movie in movie_db["movies"]:
                # print(movie)
                # print(f"query is {args.query} and movie is {movie['title']}")
                if args.query.lower().translate(punct_map) in movie[
                    "title"
                ].lower().translate(punct_map):
                    movie_hits.append(movie["title"])
            for i in range(0, len(movie_hits)):
                if i >= 5:
                    break
                print(f"{i + 1}. {movie_hits[i]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
