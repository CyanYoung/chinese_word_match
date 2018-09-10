import os

import re

import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def load_type_re(path_dir):
    word_type_re = dict()
    files = os.listdir(path_dir)
    for file in files:
        word_type = os.path.splitext(file)[0]
        word_type_re[word_type] = load_word_re(os.path.join(path_dir, file))
    return word_type_re


def replace(text, word_type_re):
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    return text


def load_pair(path):
    vocab = dict()
    for word1, word2 in pd.read_csv(path).values:
        if word1 not in vocab:
            vocab[word1] = set()
        vocab[word1].add(word2)
        if word2 not in vocab:
            vocab[word2] = set()
        vocab[word2].add(word1)
    return vocab
