from keyword_prep import remove_stopwords, prep_keywords
from pickle import dump

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def add_document(self, doc_id, text):
        tokenized_text = remove_stopwords(prep_keywords(text))
        self.index[doc_id] = tokenized_text

    def get_documents(self, term):
        doc_id_list = []

        for key in self.index.keys():
            if self.index[key] == term.lower():
                doc_id_list.append(key)

        return sorted(doc_id_list)

    def build(self, doc_id, movie_dict):
        for m in movie_dict:
            self.add_document(doc_id, f"{m['title']} {m['description']}")
            self.get_documents(term)

    def save():
