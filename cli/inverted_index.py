import pickle
from pathlib import Path

from keyword_prep import prep_keywords, remove_stopwords
from nltk.stem import PorterStemmer
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()

        tokenized_text = remove_stopwords(prep_keywords(text))
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


    def get_documents(self, term):
        # doc_id_list = []

        return sorted(list(self.index[term.lower()]))

    def get_tf(self, doc_id, term):
        stemmer = PorterStemmer()
        #print(f"term is {term}")
        if len(term) > 1:
            raise Exception ("Term Frequencies Lookup Failed due to More than One Search Term")
        else:
            term = stemmer.stem(term[0])
            #print(self.term_frequencies[doc_id])
            count = self.term_frequencies[doc_id][term]
            return count


    def build(self, movie_dict):
        for m in movie_dict["movies"]:
            movie_id = m["id"]
            self.docmap[movie_id] = m
            self.__add_document(movie_id, f"{m['title']} {m['description']}")
            # self.get_documents(term)

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

    def load(self):
        try:
            cache_dir = Path("cache")
            index_file_path = cache_dir / "index.pkl"
            docmap_file_path = cache_dir / "docmap.pkl"
            freq_file_path = cache_dir / "term_frequencies.pkl"

            with open(index_file_path, "rb") as f:
                self.index = pickle.load(f)

            with open(docmap_file_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(freq_file_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

        except:
            raise Exception("Unable to load index files")
