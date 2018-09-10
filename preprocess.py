import os

import re

from util import load_word_re, load_type_re


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def prepare(path, path_dir, mode):
    text_set = set()
    texts = list()
    labels = list()
    files = os.listdir(path_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_dir, file), 'r') as f:
            for line in f:
                text = line.strip()
                if mode == 'train':
                    text = re.sub(stop_word_re, '', text)
                    for word_type, word_re in word_type_re.items():
                        text = re.sub(word_re, word_type, text)
                if text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    labels.append(label)
    save(path, texts, labels)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_train_dir = 'data/train'
    path_test = 'data/test.csv'
    path_test_dir = 'data/test'
    prepare(path_train, path_train_dir, 'train')
    prepare(path_test, path_test_dir, 'test')
