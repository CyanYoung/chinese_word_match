import pandas as pd
import pickle as pk

import re

import numpy as np

from pypinyin import lazy_pinyin as pinyin

from Levenshtein import distance as edit_dist

from util import load_word, load_pair, list2re


path_train = 'data/train.csv'
path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
path_link = 'dict/class2word.pkl'
texts = pd.read_csv(path_train, usecols=[0]).values
stop_words = load_word(path_stop_word)
word_re = list2re(stop_words)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)
with open(path_link, 'rb') as f:
    class2word = pk.load(f)


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def select(text, match_texts):
    dists = list()
    for match_text in match_texts:
        dists.append(edit_dist(text, match_text))
    min_dist = min(dists)
    min_ind = np.argmin(np.array(dists))
    min_rate = min_dist / len(text)
    if __name__ == '__main__':
        print(text)
        print(match_texts)
        print(dists)
        print('%s %.2f' % (match_texts[int(min_ind)], min_rate))
    return min_dist, min_ind, min_rate


def predict(text):
    text = re.sub(word_re, '', text)
    phon = ''.join(pinyin(text))
    match_inds = set()
    match_texts = list()
    match_phons = list()
    match_labels = list()
    for word in text:
        cands = set()
        cands.add(word)
        find(word, cands, homo_dict)
        find(word, cands, syno_dict)
        for label in class2word.keys():
            for cand in cands:
                if cand in class2word[label]:
                    for ind in class2word[label][cand]:
                        if ind not in match_inds:
                            match_inds.add(ind)
                            match_texts.append(texts[ind][0])
                            match_phons.append(''.join(pinyin(texts[ind][0])))
                            match_labels.append(label)
    if match_texts:
        text_dist, text_ind, text_rate = select(text, match_texts)
        phon_dist, phon_ind, phon_rate = select(phon, match_phons)
        text_pred = match_labels[int(text_ind)]
        phon_pred = match_labels[int(phon_ind)]
        if text_pred == phon_pred:
            if text_rate < 0.5 or phon_rate < 0.5:
                return text_pred
            else:
                return '其它'
        else:
            if text_rate < phon_rate and text_rate < 0.5:
                return text_pred
            elif phon_rate < text_rate and phon_rate < 0.5:
                return phon_pred
            else:
                return '其它'
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('pred: %s' % predict(text))
