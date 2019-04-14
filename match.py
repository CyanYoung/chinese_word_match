import pickle as pk

import re

import numpy as np
from scipy.spatial.distance import cosine as cos_dist

from pypinyin import lazy_pinyin as pinyin

from nltk.metrics import edit_distance as edit_dist

from util import load_word_re, load_type_re, load_poly, flat_read, map_item


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def edit_predict(text, match_sents, match_labels, max_cand, thre):
    phon = ''.join(pinyin(text))
    match_phons = list()
    for sent_ind in match_sents:
        match_phons.append(''.join(pinyin(texts[sent_ind])))
    rates = list()
    for match_phon in match_phons:
        dist = edit_dist(phon, match_phon)
        rates.append(dist / len(phon))
    rates = np.array(rates)
    bound = min(len(rates), max_cand)
    min_rates = sorted(rates)[:bound]
    min_inds = np.argsort(rates)[:bound]
    min_preds = [match_labels[ind] for ind in min_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, rate, ind in zip(min_preds, min_rates, min_inds):
            formats.append('{} {:.3f} {}'.format(pred, rate, match_phons[ind]))
        return ', '.join(formats)
    if min_rates[0] < thre:
        return min_preds[0]
    else:
        return '其它'


def cos_predict(text, match_sents, match_labels, max_cand, thre):
    vec = tfidf.transform([text]).toarray()
    match_texts, sims = list(), list()
    for sent_ind, label in zip(match_sents, match_labels):
        match_texts.append(texts[sent_ind])
        match_vec = sent_vec[sent_ind]
        sims.append(1 - cos_dist(vec, match_vec))
    sims = np.array(sims)
    bound = min(len(sims), max_cand)
    max_sims = sorted(sims, reverse=True)[:bound]
    max_inds = np.argsort(-sims)[:bound]
    max_preds = [match_labels[ind] for ind in max_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, sim, ind in zip(max_preds, max_sims, max_inds):
            formats.append('{} {:.3f} {}'.format(pred, sim, match_texts[ind]))
        return ', '.join(formats)
    if max_sims[0] > thre:
        return max_preds[0]
    else:
        return '其它'


path_train = 'data/train.csv'
path_type_dir = 'dict/word_type'
path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
texts = flat_read(path_train, 'text')
word_type_re = load_type_re(path_type_dir)
stop_word_re = load_word_re(path_stop_word)
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
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    cands = set()
    for word in text:
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
        func = map_item(name, funcs)
        return func(text, match_sents, match_labels, max_cand=5, thre=0.5)
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('edit: %s' % predict(text, 'edit'))
        print('cos:  %s' % predict(text, 'cos'))
