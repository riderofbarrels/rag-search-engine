import argparse
import json
import string
from tokenize import String

from keyword_prep import prep_keywords, remove_stopwords
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            movie_hits = []
            search_tokens = remove_stopwords(prep_keywords(args.query))

            print(f"Searching for: {args.query}")
            with open("data/movies.json", "r") as f:
                movie_db = json.load(f)
            for movie in movie_db["movies"]:
                # print(movie)
                # print(f"query is {args.query} and movie is {movie['title']}")
                title_tokens = remove_stopwords(prep_keywords(movie["title"]))

                for item in search_tokens:
                    matches = [t for t in title_tokens if stemmer.stem(item) in t]
                    if matches:
                        movie_hits.append(movie["title"])
                        break

            for i in range(0, len(movie_hits)):
                if i >= 5:
                    break
                print(f"{i + 1}. {movie_hits[i]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
