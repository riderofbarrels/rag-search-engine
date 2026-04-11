from keyword_prep import remove_stopwords, prep_keywords
import pickle
from pathlib import Path

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokenized_text = text.lower().split()
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def get_documents(self, term):
        #doc_id_list = []

        return sorted(list(self.index[term.lower()]))

    def build(self, movie_dict):
        for m in movie_dict["movies"]:
            movie_id = m['id']
            self.docmap[movie_id] = m
            self.__add_document(movie_id, f"{m['title']} {m['description']}")
            #self.get_documents(term)

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / "index.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.index, f)

        file_path = cache_dir / "docmap.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.docmap, f)
