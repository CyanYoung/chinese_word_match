import os

import re
import jieba

from util import load_word, list2re


path_train_dir = 'data/train'
path_test_dir = 'data/test'
path_stop_word = 'dict/stop_word.txt'
path_special_word = 'dict/special_word.txt'
train_files = os.listdir(path_train_dir)
test_files = os.listdir(path_test_dir)
stop_words = load_word(path_stop_word)
word_re = list2re(stop_words)
jieba.load_userdict(path_special_word)


def save_file(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def prepare(path_train, path_train_cut, path_test):
    train_set = set()
    train_texts = list()
    train_cut_texts = list()
    train_labels = list()
    test_set = set()
    test_texts = list()
    test_labels = list()
    for train_file in train_files:
        label = os.path.splitext(train_file)[0]
        with open(os.path.join(path_train_dir, train_file), 'r') as f:
            for line in f:
                text = re.sub(word_re, '', line.strip())
                cut_text = ' '.join(jieba.cut(text))
                if text not in train_set:
                    train_set.add(text)
                    train_texts.append(text)
                    train_cut_texts.append(cut_text)
                    train_labels.append(label)
    save_file(path_train, train_texts, train_labels)
    save_file(path_train_cut, train_cut_texts, train_labels)
    for test_file in test_files:
        label = os.path.splitext(test_file)[0]
        with open(os.path.join(path_test_dir, test_file), 'r') as f:
            for line in f:
                text = line.strip()
                if text not in test_set:
                    test_set.add(text)
                    test_texts.append(text)
                    test_labels.append(label)
    save_file(path_test, test_texts, test_labels)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_train_cut = 'data/train_cut.csv'
    path_test = 'data/test.csv'
    prepare(path_train, path_train_cut, path_test)
