import pickle as pk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD

from util import flat_read


min_freq = 1

path_word_sent = 'feat/word_sent.pkl'
path_bow = 'model/bow.pkl'
path_svd = 'model/svd.pkl'
path_sent_vec = 'feat/sent_vec.pkl'


def link_fit(cut_texts, labels, path_word_sent):
    word_sents = dict()
    sent_ind = 0
    for cut_text, label in zip(cut_texts, labels):
        words = cut_text.split()
        for word in words:
            if word not in word_sents:
                word_sents[word] = set()
            word_sents[word].add((sent_ind, label))
        sent_ind = sent_ind + 1
    with open(path_word_sent, 'wb') as f:
        pk.dump(word_sents, f)
    if __name__ == '__main__':
        print(word_sents)


def freq_fit(cut_texts, path_bow, path_svd, path_sent_vec):
    bow = CountVectorizer(token_pattern='\w+', min_df=min_freq)
    bow.fit(cut_texts)
    bow_sents = bow.transform(cut_texts)
    svd = TruncatedSVD(n_components=200, n_iter=10)
    svd.fit(bow_sents)
    sent_vecs = svd.transform(bow_sents)
    with open(path_bow, 'wb') as f:
        pk.dump(bow, f)
    with open(path_svd, 'wb') as f:
        pk.dump(svd, f)
    with open(path_sent_vec, 'wb') as f:
        pk.dump(sent_vecs, f)


def fit(path_train):
    cut_texts = flat_read(path_train, 'cut_text')
    labels = flat_read(path_train, 'label')
    link_fit(cut_texts, labels, path_word_sent)
    freq_fit(cut_texts, path_bow, path_svd, path_sent_vec)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    fit(path_train)
