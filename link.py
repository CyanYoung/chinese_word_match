import pandas as pd
import pickle as pk

from pypinyin import lazy_pinyin


def fit(path_file, path_link, path_phon):
    word2phon = dict()
    class2word = dict()
    text_ind = 0
    for text, label in pd.read_csv(path_file).values:
        if label not in class2word:
            class2word[label] = dict()
        words = text.split(' ')
        for word in words:
            word2phon[word] = ' '.join(lazy_pinyin(word))
            if word not in class2word[label]:
                class2word[label][word] = set()
            class2word[label][word].add(text_ind)
        text_ind = text_ind + 1
    with open(path_link, 'wb') as f:
        pk.dump(class2word, f)
    with open(path_phon, 'wb') as f:
        pk.dump(word2phon, f)
    if __name__ == '__main__':
        print(word2phon)
        print(class2word)


if __name__ == '__main__':
    path_file = 'data/train.csv'
    path_link = 'dict/class2word.pkl'
    path_phon = 'dict/word2phon.pkl'
    fit(path_file, path_link, path_phon)
