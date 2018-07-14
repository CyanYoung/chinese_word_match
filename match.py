import pandas as pd
import pickle as pk

import jieba

import numpy as np

from Levenshtein import distance

from util import load_word, load_pair, filter_word


path_train = 'data/train.csv'
path_special_word = 'dict/special_word.txt'
path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
path_phon = 'dict/word2phon.pkl'
path_link = 'dict/class2word.pkl'
texts = pd.read_csv(path_train, usecols=[0]).values
jieba.load_userdict(path_special_word)
stop_words = load_word(path_stop_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)
with open(path_phon, 'rb') as f:
    word2phon = pk.load(f)
with open(path_link, 'rb') as f:
    class2word = pk.load(f)


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def predict(text):
    words = jieba.cut(text)
    valid_words = filter_word(words, stop_words)
    text = ''.join(valid_words)
    match_inds = set()
    match_texts = list()
    match_labels = list()
    for valid_word in valid_words:
        cands = set()
        cands.add(valid_word)
        find(valid_word, cands, homo_dict)
        find(valid_word, cands, syno_dict)
        for label in class2word.keys():
            for cand in cands:
                if cand in class2word[label]:
                    for ind in class2word[label][cand]:
                        if ind not in match_inds:
                            match_inds.add(ind)
                            match_texts.append(texts[ind][0].replace(' ', ''))
                            match_labels.append(label)
    edits = list()
    if match_texts:
        for match_text in match_texts:
            edits.append(distance(text, match_text))
        min_dis = min(edits)
        min_ind = np.argmin(np.array(edits))
        if __name__ == '__main__':
            print(text)
            print(match_texts)
            print(edits)
            print(match_texts[int(min_ind)])
        if min_dis > 1:
            return '其它'
        else:
            return match_labels[int(min_ind)]
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('pred: %s' % predict(text))
