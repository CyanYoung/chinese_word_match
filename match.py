import pandas as pd
import pickle as pk

import re

import numpy as np

from pypinyin import lazy_pinyin as pinyin

from nltk.metrics import edit_distance as edit_dist

from util import load_word_re, load_type_re, load_poly, map_item


path_train = 'data/train.csv'
path_type_dir = 'dict/word_type'
path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
texts = pd.read_csv(path_train, usecols=['text']).values
word_type_re = load_type_re(path_type_dir)
stop_word_re = load_word_re(path_stop_word)
homo_dict = load_poly(path_homo)
syno_dict = load_poly(path_syno)

path_class2word = 'feat/class2word.pkl'
path_tfidf = 'model/tfidf.pkl'
path_ind2vec = 'feat/ind2vec.pkl'
with open(path_class2word, 'rb') as f:
    class2word = pk.load(f)
with open(path_tfidf, 'rb') as f:
    tfidf = pk.load(f)
with open(path_ind2vec, 'rb') as f:
    ind2vec = pk.load(f)


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def edit_predict(text, match_inds, match_labels):
    phon = ''.join(pinyin(text))
    match_phons = list()
    for match_ind in match_inds:
        match_phons.append(''.join(pinyin(texts[match_ind][0])))
    rates = list()
    for match_phon in match_phons:
        dist = edit_dist(phon, match_phon)
        rates.append(dist / len(phon))
    min_rate = min(rates)
    min_ind = np.argmin(np.array(rates))
    if __name__ == '__main__':
        formats = list()
        bound = min(len(match_phons), 5)
        for match_phon, rate in zip(match_phons[:bound], rates[:bound]):
            formats.append('{} {:.3f}'.format(match_phon, rate))
        print(', '.join(formats))
        print('{} {:.3f}'.format(match_phons[int(min_ind)], min_rate))
    if min_rate < 0.5:
        return match_labels[int(min_ind)]
    else:
        return '其它'


def cos_sim(vec1, vec2):
    deno = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if deno:
        return (np.dot(vec1, vec2) / deno)[0]
    else:
        return 0.0


def cos_predict(text, match_inds, match_labels):
    vec = dict()
    for label, model in tfidf.items():
        vec[label] = model.transform([text]).toarray()
    match_texts = list()
    for match_ind in match_inds:
        match_texts.append(texts[match_ind][0])
    sims = list()
    for match_ind, match_label in zip(match_inds, match_labels):
        match_vec = ind2vec[match_ind]
        sims.append(cos_sim(vec[match_label], match_vec))
    max_sim = max(sims)
    max_ind = np.argmax(np.array(sims))
    if __name__ == '__main__':
        formats = list()
        bound = min(len(match_texts), 5)
        for match_text, sim in zip(match_texts[:bound], sims[:bound]):
            formats.append('{} {:.3f}'.format(match_text, sim))
        print(', '.join(formats))
        print('{} {:.3f}'.format(match_texts[int(max_ind)], max_sim))
    if max_sim > 0.5:
        return match_labels[int(max_ind)]
    else:
        return '其它'


funcs = {'edit': edit_predict,
         'cos': cos_predict}


def predict(text, name):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    ind_set = set()
    match_inds = list()
    match_labels = list()
    for word in text:
        cands = set()
        cands.add(word)
        find(word, cands, homo_dict)
        find(word, cands, syno_dict)
        for label, words in class2word.items():
            for cand in cands:
                if cand in words:
                    for ind in words[cand]:
                        if ind not in ind_set:
                            ind_set.add(ind)
                            match_inds.append(ind)
                            match_labels.append(label)
    if match_inds:
        func = map_item(name, funcs)
        return func(text, match_inds, match_labels)
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('edit: %s' % predict(text, 'edit'))
        print('cos:  %s' % predict(text, 'cos'))
