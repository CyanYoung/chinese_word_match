import pickle as pk

from sklearn.feature_extraction.text import TfidfVectorizer

from util import flat_read


min_freq = 1

path_word_sent = 'feat/word_sent.pkl'
path_tfidf = 'model/tfidf.pkl'
path_sent_vec = 'feat/sent_vec.pkl'


def link_fit(texts, labels, path_word_sent):
    word_sents = dict()
    sent_ind = 0
    for text, label in zip(texts, labels):
        for word in text:
            if word not in word_sents:
                word_sents[word] = set()
            word_sents[word].add((sent_ind, label))
        sent_ind = sent_ind + 1
    with open(path_word_sent, 'wb') as f:
        pk.dump(word_sents, f)
    if __name__ == '__main__':
        print(word_sents)


def freq_fit(texts, labels, path_tfidf, path_sent_vec):
    label2text = dict()
    for text, label in zip(texts, labels):
        if label not in label2text:
            label2text[label] = list()
        label2text[label].append(text)
    docs = list()
    for texts in label2text.values():
        docs.append(''.join(texts))
    model = TfidfVectorizer(token_pattern='\w', min_df=min_freq)
    model.fit(texts)
    sent_vecs = model.transform(texts).toarray()
    with open(path_tfidf, 'wb') as f:
        pk.dump(model, f)
    with open(path_sent_vec, 'wb') as f:
        pk.dump(sent_vecs, f)


def fit(path_train):
    texts = flat_read(path_train, 'text')
    labels = flat_read(path_train, 'label')
    link_fit(texts, labels, path_word_sent)
    freq_fit(texts, labels, path_tfidf, path_sent_vec)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    fit(path_train)
