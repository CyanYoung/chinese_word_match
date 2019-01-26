import pickle as pk

from sklearn.feature_extraction.text import TfidfVectorizer

from util import flat_read


min_freq = 1

path_label2word = 'feat/label2word.pkl'
path_tfidf = 'model/tfidf.pkl'
path_ind2vec = 'feat/ind2vec.pkl'


def link_fit(texts, labels, path_label2word):
    label2word = dict()
    ind = 0
    for text, label in zip(texts, labels):
        if label not in label2word:
            label2word[label] = dict()
        for word in text:
            if word not in label2word[label]:
                label2word[label][word] = set()
            label2word[label][word].add(ind)
        ind = ind + 1
    with open(path_label2word, 'wb') as f:
        pk.dump(label2word, f)
    if __name__ == '__main__':
        print(label2word)


def freq_fit(texts, labels, path_tfidf, path_ind2vec):
    label2text, label2ind = dict(), dict()
    ind = 0
    for text, label in zip(texts, labels):
        if label not in label2text:
            label2text[label], label2ind[label] = list(), list()
        label2text[label].append(text)
        label2ind[label].append(ind)
        ind = ind + 1
    tfidf, ind2vec = dict(), dict()
    for label, texts in label2text.items():
        tfidf[label] = TfidfVectorizer(token_pattern='\w', min_df=min_freq)
        tfidf[label].fit(texts)
        vecs = tfidf[label].transform(texts).toarray()
        inds = label2ind[label]
        for ind, vec in zip(inds, vecs):
            ind2vec[ind] = vec
    with open(path_tfidf, 'wb') as f:
        pk.dump(tfidf, f)
    with open(path_ind2vec, 'wb') as f:
        pk.dump(ind2vec, f)
    if __name__ == '__main__':
        print(label2text)


def fit(path_train):
    texts = flat_read(path_train, 'text')
    labels = flat_read(path_train, 'label')
    link_fit(texts, labels, path_label2word)
    freq_fit(texts, labels, path_tfidf, path_ind2vec)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    fit(path_train)
