import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def filter_word(words, stop_words):
    valid_words = list()
    for word in words:
        if word not in stop_words:
            valid_words.append(word)
    return valid_words


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
