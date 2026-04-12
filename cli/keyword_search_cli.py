import argparse
import json

from inverted_index import InvertedIndex
from keyword_prep import prep_keywords, remove_stopwords
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser(
        "build", help="Buid an inverted index of Movies"
    )

    args = parser.parse_args()
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            search_tokens = remove_stopwords(prep_keywords(args.query))

            print(f"Searching for: {args.query}")

            numhits = 0
            doc_hits = []

            # iterate through the search tokens and return up to 5 hits from the inverted index
            for item in search_tokens:
                if stemmer.stem(item) in ii.index.keys():
                    print(f"hit found for {item}")
                    doc_indices = ii.get_documents(item)
                    for hit in doc_indices:
                        # print(f"Trying to get doc for index {hit}")
                        doc_hits.append(ii.docmap[hit])
                        numhits += 1
                        # print(f"updated numhits to {numhits}")
                        if numhits == 5:
                            # print("doc limit reached. ending search.")
                            break
                if numhits == 5:
                    # print("completed search before hitting limit. ending search")
                    break

            # print the 5 ids/titles
            for hit in doc_hits:
                print(f"{hit['id']}: {hit['title']}")

        #            with open("data/movies.json", "r") as f:
        #                movie_db = json.load(f)
        #            for movie in movie_db["movies"]:
        #                # print(movie)
        #                # print(f"query is {args.query} and movie is {movie['title']}")
        #                title_tokens = remove_stopwords(prep_keywords(movie["title"]))

        #               for item in search_tokens:
        #                    matches = [t for t in title_tokens if stemmer.stem(item) in t]
        #                    if matches:
        #                        movie_hits.append(movie["title"])
        #                        break

        #            for i in range(0, len(movie_hits)):
        #                if i >= 5:
        #                    break
        #                print(f"{i + 1}. {movie_hits[i]}")

        case "build":
            with open("data/movies.json", "r") as f:
                movie_db = json.load(f)
            ii = InvertedIndex()
            ii.build(movie_db)
            ii.save()

            # docs = ii.get_documents("merida")
            # print(docs[0])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
