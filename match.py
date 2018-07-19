import pandas as pd
import pickle as pk

import re
import jieba

import numpy as np

from pypinyin import lazy_pinyin as pinyin

from Levenshtein import distance as edit_dist

from util import load_word, load_pair, list2re


path_train = 'data/train.csv'
path_stop_word = 'dict/stop_word.txt'
path_special_word = 'dict/special_word.txt'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
path_class2word = 'dict/class2word.pkl'
path_tfidf = 'model/tfidf.pkl'
path_class2doc = 'dict/class2doc.pkl'
texts = pd.read_csv(path_train, usecols=[0]).values
stop_words = load_word(path_stop_word)
word_re = list2re(stop_words)
jieba.load_userdict(path_special_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)
with open(path_class2word, 'rb') as f:
    class2word = pk.load(f)
with open(path_tfidf, 'rb') as f:
    tfidf = pk.load(f)
with open(path_class2doc, 'rb') as f:
    class2doc = pk.load(f)


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def select(phon, match_phons):
    dists = list()
    for match_phon in match_phons:
        dists.append(edit_dist(phon, match_phon))
    min_dist = min(dists)
    min_ind = np.argmin(np.array(dists))
    min_rate = min_dist / len(phon)
    if __name__ == '__main__':
        print(phon)
        print(match_phons)
        print(dists)
        print('%s %.2f' % (match_phons[int(min_ind)], min_rate))
    return min_dist, min_ind, min_rate


def edit_predict(text):
    phon = ''.join(pinyin(text))
    match_inds = set()
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
                            match_phons.append(''.join(pinyin(texts[ind][0])))
                            match_labels.append(label)
    if match_phons:
        min_dist, min_ind, min_rate = select(phon, match_phons)
        if min_rate < 0.3:
            return match_labels[int(min_ind)]
        else:
            return '其它'
    else:
        return '其它'


def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def cos_predict(cut_text):
    text_vec = tfidf.transform([cut_text]).toarray()
    sims = list()
    labels = list()
    for label in class2doc.keys():
        sims.append(cos_sim(text_vec, class2doc[label])[0])
        labels.append(label)
    max_sim = max(sims)
    max_ind = np.argmax(np.array(sims))
    if __name__ == '__main__':
        print(cut_text)
        print(sims)
    if max_sim > 0.1:
        return labels[int(max_ind)]
    else:
        return '其它'


def predict(text, metric):
    text = re.sub(word_re, '', text)
    if metric == 'edit_dist':
        return edit_predict(text)
    elif metric == 'cos_sim':
        cut_text = ' '.join(jieba.cut(text))
        return cos_predict(cut_text)
    else:
        raise KeyError


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('edit_pred: %s' % predict(text, 'edit_dist'))
        print('cos_pred: %s' % predict(text, 'cos_sim'))
