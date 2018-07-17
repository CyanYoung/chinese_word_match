import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def list2re(words):
    word_re = '['
    for word in words:
        word_re = word_re + '(' + word + ')'
    return word_re + ']'


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
