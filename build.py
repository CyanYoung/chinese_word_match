import pandas as pd
import pickle as pk

from sklearn.feature_extraction.text import TfidfVectorizer


min_freq = 1

path_class2word = 'dict/class2word.pkl'
path_tfidf = 'model/tfidf.pkl'
path_ind2vec = 'dict/ind2vec.pkl'


def link_fit(path_train, path_class2word):
    class2word = dict()
    ind = 0
    for text, label in pd.read_csv(path_train).values:
        if label not in class2word:
            class2word[label] = dict()
        for word in text:
            if word not in class2word[label]:
                class2word[label][word] = set()
            class2word[label][word].add(ind)
        ind = ind + 1
    with open(path_class2word, 'wb') as f:
        pk.dump(class2word, f)
    if __name__ == '__main__':
        print(class2word)


def freq_fit(path_train, path_tfidf, path_ind2vec):
    class2text = dict()
    class2ind = dict()
    ind = 0
    for text, label in pd.read_csv(path_train).values:
        if label not in class2text:
            class2text[label] = list()
            class2ind[label] = list()
        class2text[label].append(text)
        class2ind[label].append(ind)
        ind = ind + 1
    tfidf = dict()
    ind2vec = dict()
    for label, texts in class2text.items():
        tfidf[label] = TfidfVectorizer(token_pattern='\w', min_df=min_freq)
        tfidf[label].fit(texts)
        vecs = tfidf[label].transform(texts).toarray()
        inds = class2ind[label]
        for ind, vec in zip(inds, vecs):
            ind2vec[ind] = vec
    with open(path_tfidf, 'wb') as f:
        pk.dump(tfidf, f)
    with open(path_ind2vec, 'wb') as f:
        pk.dump(ind2vec, f)
    if __name__ == '__main__':
        print(class2text)


def fit(path_train):
    link_fit(path_train, path_class2word)
    freq_fit(path_train, path_tfidf, path_ind2vec)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    fit(path_train)
