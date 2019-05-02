import os

import re
import jieba

from random import shuffle

from util import load_word_re, load_type_re


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)

path_cut_word = 'dict/cut_word.txt'
jieba.load_userdict(path_cut_word)


def save(path, texts, cut_texts, labels):
    head = 'text,cut_text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, cut_text, label in zip(texts, cut_texts, labels):
            f.write(text + ',' + cut_text + ',' + label + '\n')


def clean(text):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    return text


def prepare(path_univ_dir, path_train, path_test):
    text_set = set()
    texts, cut_texts, labels = list(), list(), list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = clean(text)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    cut_text = ' '.join(jieba.cut(text))
                    cut_texts.append(cut_text)
                    labels.append(label)
    texts_labels = list(zip(texts, cut_texts, labels))
    shuffle(texts_labels)
    texts, cut_texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts, cut_texts[:bound], labels[:bound])
    save(path_test, texts, cut_texts[bound:], labels[bound:])


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    prepare(path_univ_dir, path_train, path_test)

