import string
from tracemalloc import stop


def prep_keywords(keywords):
    punct_map = str.maketrans("", "", string.punctuation)
    clean_list = keywords.lower().translate(punct_map).split()
    return clean_list


def remove_stopwords(keywords):
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    return list(set(keywords) - set(stopwords))
