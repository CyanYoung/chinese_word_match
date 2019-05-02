import pickle as pk

import jieba

import numpy as np
from scipy.spatial.distance import cosine as cos_dist

from pypinyin import lazy_pinyin as pinyin

from nltk.metrics import edit_distance as edit_dist

from preprocess import clean

from util import load_poly, flat_read


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def sort(dists, match_texts, match_labels, max_cand, thre):
    dists = np.array(dists)
    bound = min(len(dists), max_cand)
    min_dists = sorted(dists)[:bound]
    min_inds = np.argsort(dists)[:bound]
    min_preds = [match_labels[ind] for ind in min_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, rate, ind in zip(min_preds, min_dists, min_inds):
            formats.append('{} {:.3f} {}'.format(pred, rate, match_texts[ind]))
        return ', '.join(formats)
    if min_dists[0] < thre:
        return min_preds[0]
    else:
        return '其它'


def edit_predict(text, match_sents, match_labels, max_cand, thre):
    phon = ''.join(pinyin(text))
    match_phons = list()
    for sent_ind in match_sents:
        match_phons.append(''.join(pinyin(texts[sent_ind])))
    rates = list()
    for match_phon in match_phons:
        dist = edit_dist(phon, match_phon)
        rates.append(dist / len(phon))
    return sort(rates, match_phons, match_labels, max_cand, thre)


def cos_predict(cut_text, match_sents, match_labels, max_cand, thre):
    vec = tfidf.transform([cut_text]).toarray()
    match_texts, dists = list(), list()
    for sent_ind, label in zip(match_sents, match_labels):
        match_texts.append(texts[sent_ind])
        match_vec = sent_vec[sent_ind]
        dists.append(cos_dist(vec, match_vec))
    return sort(dists, match_texts, match_labels, max_cand, thre)


path_cut_word = 'dict/cut_word.txt'
jieba.load_userdict(path_cut_word)

path_train = 'data/train.csv'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
texts = flat_read(path_train, 'text')
homo_dict = load_poly(path_homo)
syno_dict = load_poly(path_syno)

path_word_sent = 'feat/word_sent.pkl'
path_tfidf = 'model/tfidf.pkl'
path_sent_vec = 'feat/sent_vec.pkl'
with open(path_word_sent, 'rb') as f:
    word_sent = pk.load(f)
with open(path_tfidf, 'rb') as f:
    tfidf = pk.load(f)
with open(path_sent_vec, 'rb') as f:
    sent_vec = pk.load(f)

funcs = {'edit': edit_predict,
         'cos': cos_predict}


def predict(text, name):
    text = clean(text)
    cut_text = ' '.join(jieba.cut(text))
    words = cut_text.split()
    cands = set()
    for word in words:
        if word not in cands:
            cands.add(word)
            find(word, cands, homo_dict)
            find(word, cands, syno_dict)
    ind_set = set()
    match_sents, match_labels = list(), list()
    for cand in cands:
        if cand in word_sent:
            pairs = word_sent[cand]
            for sent_ind, label in pairs:
                if sent_ind not in ind_set:
                    ind_set.add(sent_ind)
                    match_sents.append(sent_ind)
                    match_labels.append(label)
    if match_sents:
        if name == 'edit':
            return edit_predict(text, match_sents, match_labels, max_cand=5, thre=0.8)
        else:
            return cos_predict(cut_text, match_sents, match_labels, max_cand=5, thre=0.8)
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('edit: %s' % predict(text, 'edit'))
        print('cos:  %s' % predict(text, 'cos'))
