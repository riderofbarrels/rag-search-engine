import argparse
import json
import math

from inverted_index import BM25_B, BM25_K1, InvertedIndex
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

    tf_parser = subparsers.add_parser(
        "tf", help="Get the number of times a term appears in a movie entry"
    )
    tf_parser.add_argument("doc_id", type=int, help="ID of document to query")
    tf_parser.add_argument("search_term", type=str, help="The term to search for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverse frequency of the search term"
    )
    idf_parser.add_argument("search_term", type=str, help="The term to find the IDF of")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Gets the Term Frequency Inverse Document Frequency"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="ID of document to query")
    tfidf_parser.add_argument(
        "search_term", type=str, help="The term to find the TF-IDF of"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
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

        case "build":
            with open("data/movies.json", "r") as f:
                movie_db = json.load(f)
            ii = InvertedIndex()
            ii.build(movie_db)
            ii.save()

        case "tf":
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            search_term = remove_stopwords(prep_keywords(args.search_term))
            doc_id = args.doc_id

            print(ii.get_tf(doc_id, search_term))
        case "idf":
            # loads the index, docmap, and freqency objects
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            # clean the search term
            search_term = remove_stopwords(prep_keywords(args.search_term))
            search_term = stemmer.stem(search_term[0])

            # get the count by checking the length of matched doc_ids in the index, the count # of docs in docmap, then calculate IDF
            docs_with_term_count = len(ii.index[search_term])
            total_doc_count = len(ii.docmap)
            idf = math.log((total_doc_count + 1) / (docs_with_term_count + 1))

            print(f"{idf: .2f}")

        case "tfidf":
            # loads the index, docmap, and freqency objects
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            # clean the search term and store the doc_id to search against
            search_term = remove_stopwords(prep_keywords(args.search_term))
            stemmer.stem(search_term[0])
            doc_id = args.doc_id

            # get the term frequency and inverse document frequency
            tf = ii.get_tf(doc_id, search_term)
            docs_with_term_count = len(ii.index[stemmer.stem(search_term[0])])
            total_doc_count = len(ii.docmap)
            idf = math.log((total_doc_count + 1) / (docs_with_term_count + 1))

            # Get the tf-IDF
            tfidf = tf * idf
            print(
                f"TF-IDF score of '{args.search_term}' in document '{args.doc_id}': {tfidf:.2f}"
            )

        case "bm25idf":
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            # clean the search term and store the doc_id to search against
            search_term = remove_stopwords(prep_keywords(args.term))
            search_term = stemmer.stem(search_term[0])
            print(search_term)
            bm25idf = ii.get_bm25_idf(search_term)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            ii = InvertedIndex()
            try:
                ii.load()
            except:
                raise Exception("Failed to load index and docmap")

            search_term = remove_stopwords(prep_keywords(args.term))
            stemmer.stem(search_term[0])
            doc_id = args.doc_id
            bm25tf = ii.get_bm25_tf(doc_id, search_term, k1=BM25_K1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
