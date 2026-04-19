import pickle
from collections import Counter
from math import log
from pathlib import Path
from statistics import mean

from keyword_prep import prep_keywords, remove_stopwords
from nltk.stem import PorterStemmer

BM25_K1 = 1.5  # the constant which controls the saturation effect in the BM25 saturation formula
BM25_B = 0.75
LIMIT = 5


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()
        # prep/tokenize text, but do not remove stopwords
        tokenized_text = prep_keywords(text)

        # Count the token after cleanup and store in the doc_lengths dict
        token_count = len(tokenized_text)
        self.doc_lengths.update({doc_id: token_count})

        # remove stopwords from tokens
        tokenized_text = remove_stopwords(tokenized_text)

        for token in tokenized_text:
            stemmed_token = stemmer.stem(token)
            if stemmed_token not in self.index:
                self.index[stemmed_token] = []
            if doc_id not in self.index[stemmed_token]:
                self.index[stemmed_token].append(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
                self.term_frequencies[doc_id][stemmed_token] = 0
            self.term_frequencies[doc_id][stemmed_token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0
        return mean(self.doc_lengths.values())

    def get_documents(self, term):
        # doc_id_list = []

        return sorted(list(self.index[term.lower()]))

    def get_tf(self, doc_id, term):
        stemmer = PorterStemmer()
        # print(f"term is {term}")
        # if len(term) > 1:
        #    raise Exception(
        #        "Term Frequencies Lookup Failed due to More than One Search Term"
        #    )
        # else:
        #    term = stemmer.stem(term[0])
        # print(self.term_frequencies[doc_id])
        count = self.term_frequencies[doc_id][term]
        return count

    def get_bm25_idf(self, term: str) -> float:
        stemmer = PorterStemmer()
        # print(f"getting bm25 for {stemmer.stem(term)}")
        num_docs = len(self.docmap)
        num_docs_w_term = len(self.index[stemmer.stem(term)])
        bm25_idf = log((num_docs - num_docs_w_term + 0.5) / (num_docs_w_term + 0.5) + 1)

        return bm25_idf

    def build(self, movie_dict):
        for m in movie_dict["movies"]:
            movie_id = m["id"]
            self.docmap[movie_id] = m
            self.__add_document(movie_id, f"{m['title']} {m['description']}")
            # self.get_documents(term)
            #

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=LIMIT):
        score_dict = {}
        tokenized_query = remove_stopwords(prep_keywords(query))

        for doc_id in self.docmap.keys():
            score_dict[doc_id] = 0
            for term in tokenized_query:
                term_bm25 = self.bm25(doc_id, term)
                score_dict[doc_id] += term_bm25

            # sort by descending score
            # retun the top LIMIT items

        return score_dict

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / "index.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.index, f)

        file_path = cache_dir / "docmap.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.docmap, f)

        file_path = cache_dir / "term_frequencies.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        file_path = cache_dir / "doc_lengths.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
        print(self.doc_lengths)

    def load(self):
        try:
            cache_dir = Path("cache")
            index_file_path = cache_dir / "index.pkl"
            docmap_file_path = cache_dir / "docmap.pkl"
            freq_file_path = cache_dir / "term_frequencies.pkl"
            lengths_file_path = cache_dir / "doc_lengths.pkl"

            with open(index_file_path, "rb") as f:
                self.index = pickle.load(f)

            with open(docmap_file_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(freq_file_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

            with open(lengths_file_path, "rb") as f:
                self.doc_lengths = pickle.load(f)

        except:
            raise Exception("Unable to load index files")
