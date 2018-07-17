import pandas as pd
import pickle as pk


def fit(path_file, path_link):
    class2word = dict()
    text_ind = 0
    for text, label in pd.read_csv(path_file).values:
        if label not in class2word:
            class2word[label] = dict()
        for word in text:
            if word not in class2word[label]:
                class2word[label][word] = set()
            class2word[label][word].add(text_ind)
        text_ind = text_ind + 1
    with open(path_link, 'wb') as f:
        pk.dump(class2word, f)
    if __name__ == '__main__':
        print(class2word)


if __name__ == '__main__':
    path_file = 'data/train.csv'
    path_link = 'dict/class2word.pkl'
    fit(path_file, path_link)
