import pandas as pd
import pickle as pk

from sklearn.feature_extraction.text import TfidfVectorizer


def edit_fit(path_train, path_class2word):
    class2word = dict()
    text_ind = 0
    for text, label in pd.read_csv(path_train).values:
        if label not in class2word:
            class2word[label] = dict()
        for word in text:
            if word not in class2word[label]:
                class2word[label][word] = set()
            class2word[label][word].add(text_ind)
        text_ind = text_ind + 1
    with open(path_class2word, 'wb') as f:
        pk.dump(class2word, f)
    if __name__ == '__main__':
        print(class2word)


def cos_fit(path_train, path_tfidf, path_class2doc):
    class2doc = dict()
    docs = list()
    labels = list()
    for text, label in pd.read_csv(path_train).values:
        if label not in class2doc:
            class2doc[label] = ''
        class2doc[label] = class2doc[label] + text
    for label in class2doc.keys():
        docs.append(class2doc[label])
        labels.append(label)
    tfidf = TfidfVectorizer(token_pattern='\w', min_df=1)
    tfidf.fit(docs)
    feature = tfidf.transform(docs).toarray()
    if __name__ == '__main__':
        print(class2doc)
    for i in range(len(labels)):
        class2doc[labels[i]] = feature[i]
    with open(path_tfidf, 'wb') as f:
        pk.dump(tfidf, f)
    with open(path_class2doc, 'wb') as f:
        pk.dump(class2doc, f)


def fit(path_train, path_class2word, path_tfidf, path_class2doc):
    edit_fit(path_train, path_class2word)
    cos_fit(path_train, path_tfidf, path_class2doc)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_class2word = 'dict/class2word.pkl'
    path_tfidf = 'model/tfidf.pkl'
    path_class2doc = 'dict/class2doc.pkl'
    fit(path_train, path_class2word, path_tfidf, path_class2doc)
