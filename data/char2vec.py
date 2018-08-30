#-*- coding:utf-8 -*-
import sys
import os
import random
import numpy as np

import multiprocessing
from multiprocessing import Pool
import gensim
from gensim.corpora import TextCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

np.random.seed(10)

PAD_ID = 0
GO_ID = 1
END_ID = 2
NUM_ALP_ID = 3
UNK_ID = 4

char2vec_format_corpus = 'char2vec_train.txt'
c2v_npy_file = 'char_vec.npy'
vocab_file = 'vocab.txt'


def train_char2vec(c2v_dim=100):
	print 'Training char2vec...'
	model = Word2Vec(LineSentence(char2vec_format_corpus), size=c2v_dim, window=5, min_count=5, workers=multiprocessing.cpu_count())
	print 'End...'

	count = 0
	char_vec_matrix = np.zeros((len(open(vocab_file, 'r').readlines()), c2v_dim), dtype='float')

	for idx, line in enumerate(open(vocab_file, 'r').readlines()):
		if idx == 0: # PAD
			char_vec_matrix[idx, :] = np.zeros((1, c2v_dim), dtype='float')
			continue
		c = line.strip()
		try:
			char_vec = model.wv[unicode(c)]
			char_vec_matrix[idx, :] = char_vec
			print c
		except:
			count += 1
			char_vec_matrix[idx, :] = np.random.uniform(-1.0, 1.0, size=(1, c2v_dim))

	print count
	np.save(c2v_npy_file, char_vec_matrix)


if __name__ == "__main__":
	train_char2vec(100)
