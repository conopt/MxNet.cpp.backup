#!/usr/bin/env python3
import numpy
import pickle
import os

vocab_size = 20000

def trim_vocab(p_vocab, p_vocab_size=40000):
    return dict(kv for kv in p_vocab.items() if kv[1] < p_vocab_size)

def trim_unk(seq, vocab):
    return [w if w in vocab else '<UNK>' for w in seq]

def seq2id(seq, vocab):
    return [str(vocab[w]) for w in seq]

def seq2emb(seq, embedding):
    shape = next(iter(embedding.values())).shape
    zero = numpy.zeros(shape)
    return [embedding.get(w, zero) for w in seq]

def build_vocab(p_path_to_data, p_processor=lambda l:l.rstrip()):
    vocab_freq = {}
    vocab = { '<PAD>':0, '<S>':1, '</S>':2, '<UNK>':3 }
    next_idx = 4
    for l in open(p_path_to_data, 'r', encoding='utf-8'):
        l = p_processor(l)
        for w in set(l.split()):
            if w not in vocab_freq:
                vocab_freq[w] = 0
            vocab_freq[w] += 1
    
    for w, freq in sorted(vocab_freq.items(), key=lambda kv: kv[1], reverse=True):
        vocab[w] = next_idx
        next_idx += 1
    return vocab

data_path = 'dcs_url_title_query_norm_top10k.txt'
embedding_path = 'vectors.txt.pkl'
vocab_path = 'vocab.pkl'

if not os.path.isfile(vocab_path):
    print('Building vocab...')
    vocab = build_vocab(data_path, p_processor=lambda l:' '.join(l.split('\t'))[:3])
    with open(vocab_path, 'wb') as fo:
        pickle.dump(vocab, fo, protocol=4)
else:
    print('Loading vocab...')
    vocab = pickle.load(open(vocab_path, 'rb'))

if len(vocab) > vocab_size:
    print('Trimming vocab...')
    vocab = trim_vocab(vocab, vocab_size)

print('Loading embeddings')
embedding = pickle.load(open(embedding_path, 'rb'))

x_path = 'query_id.txt'
y_path = 'answer_id.txt'
x_emb_path = 'query_emb.bin'
y_emb_path = 'answer_emb.bin'
x = open(x_path, 'w')
y = open(y_path, 'w')
x_emb = open(x_emb_path, 'wb')
y_emb = open(y_emb_path, 'wb')
dim = str(next(iter(embedding.values())).shape[0])+'\n'
x_emb.write(dim.encode('ascii'))
y_emb.write(dim.encode('ascii'))

print('Preprocessing inputs')
with open(data_path) as f:
    lines = f.readlines()
    for line in lines:
        cols = line.split('\t')
        src = trim_unk(cols[0].split(' '), vocab)
        tar = trim_unk(cols[1].split(' '), vocab)
        x.write(' '.join(seq2id(src, vocab)) + '\n')
        y.write(' '.join(seq2id(tar, vocab)) + '\n')
        x_emb.write(numpy.concatenate(seq2emb(src, embedding)).tostring())
        y_emb.write(numpy.concatenate(seq2emb(tar, embedding)).tostring())
