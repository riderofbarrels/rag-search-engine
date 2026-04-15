import string


def prep_keywords(keywords):
    punct_map = str.maketrans("", "", string.punctuation)
    clean_list = keywords.lower().translate(punct_map).split()
    return clean_list


def remove_stopwords(keywords):
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    # create a set of keywords with stopwords removed so that we can use it to filter the full list of keywords
    valid_words = list(set(keywords) - set(stopwords))
    # filter the keywords to retain only valid keywords
    keywords = list(filter(lambda a: a in valid_words, keywords))
    return keywords
