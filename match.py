import pickle as pk

import re

import numpy as np

from pypinyin import lazy_pinyin as pinyin

from nltk.metrics import edit_distance as edit_dist

from util import load_word_re, load_type_re, load_poly, flat_read, map_item


def find(word, cands, word_dict):
    if word in word_dict:
        for cand in word_dict[word]:
            cands.add(cand)


def edit_predict(sent, texts, match_inds, match_labels, cand, thre):
    phon = ''.join(pinyin(sent))
    match_phons = list()
    for ind in match_inds:
        match_phons.append(''.join(pinyin(texts[ind])))
    rates = list()
    for match_phon in match_phons:
        dist = edit_dist(phon, match_phon)
        rates.append(dist / len(phon))
    rates = np.array(rates)
    bound = min(len(rates), cand)
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


def cos_sim(vec1, vec2):
    deno = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if deno:
        return (np.dot(vec1, vec2) / deno)[0]
    else:
        return 0.0


def cos_predict(sent, texts, match_inds, match_labels, cand, thre):
    vecs = dict()
    for label, model in tfidf.items():
        vecs[label] = model.transform([sent]).toarray()
    match_texts = list()
    for ind in match_inds:
        match_texts.append(texts[ind])
    sims = list()
    for ind, label in zip(match_inds, match_labels):
        match_vec = ind2vec[ind]
        sims.append(cos_sim(vecs[label], match_vec))
    sims = np.array(sims)
    bound = min(len(sims), cand)
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
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
texts = flat_read(path_train, 'text')
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

funcs = {'edit': edit_predict,
         'cos': cos_predict}


def predict(text, name):
    sent = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        sent = re.sub(word_re, word_type, sent)
    ind_set = set()
    match_inds = list()
    match_labels = list()
    for word in sent:
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
        return func(sent, texts, match_inds, match_labels, cand=5, thre=0.5)
    else:
        return '其它'


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('edit: %s' % predict(text, 'edit'))
        print('cos:  %s' % predict(text, 'cos'))
