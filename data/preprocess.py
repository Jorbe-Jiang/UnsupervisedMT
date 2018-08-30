#-*- coding:utf-8 -*-
import sys
import os
import random
import numpy as np

import collections
from collections import Counter
import math

import re

np.random.seed(10)

# sen | source(0/1)
output_train_file = 'format_train.data'
output_valid_file = 'format_valid.data'
output_test_file = 'format_test.data'

# sen_1 | sen_2
output_mt_train_file = 'format_mt_train.data'
output_mt_valid_file = 'format_mt_valid.data'
output_mt_test_file = 'format_mt_test.data'

# for training char vector
char2vec_train_corpus = 'char2vec_train.txt'

# slot name dict
slot_name_dict_file = 'slot_name_dict.txt'

c2v_npy_file = 'char_vec.npy'
vocab_file = 'vocab.txt'
speech_dir = './speech/'

PAD_ID = 0
GO_ID = 1
END_ID = 2
NUM_ALP_ID = 3
UNK_ID = 4


def filter_line(line):
	line = line.decode("utf-8")
	cleaned_txt = re.sub(
		"[\s+\\/_$%^*(){}+\"\']+|[+——~@#￥%……&*（）⊙○\|\[\]=／ⅢⅡ\－\–の≠■◎□↑‘’＂•·×●②①〓\-“”《》【】¶]+".decode(
			"utf8"), "".decode("utf8"), line)
	return cleaned_txt


def get_files_name(speech_dir):
	files_name = os.listdir(speech_dir)
	return files_name


def preprocess_speech():
	files_name = get_files_name(speech_dir)
	train_fp = open(output_train_file, 'w')
	valid_fp = open(output_valid_file, 'w')
	test_fp = open(output_test_file, 'w')

	train_mt_fp = open(output_mt_train_file, 'w')
	valid_mt_fp = open(output_mt_valid_file, 'w')
	test_mt_fp = open(output_mt_test_file, 'w')

	char2vcec_fp = open(char2vec_train_corpus, 'w')
	vocab_fp = open(vocab_file, 'w')
	slot_fp = open(slot_name_dict_file, 'w')

	vocab = []
	slot_name_dict = {}
	name_start_idx = 0

	for file_name in files_name:
		if '~' in file_name:
			continue
		with open(speech_dir + file_name, 'r') as fp:
			for idx, line in enumerate(fp.readlines()):
				sen_li = line.strip().split()
				sen_li = [filter_line(sen_li[0].strip()), filter_line(sen_li[1].strip())]
				sen_1_cut, sen_1_slot_name_li, sen_1_slot_name_idx_li, name_start_idx = cut_words(sen_li[0].strip(), slot_name_dict, name_start_idx)

				for idx, slot_name in enumerate(sen_1_slot_name_li):
					slot_name_dict[slot_name] = sen_1_slot_name_idx_li[idx]

				for char in sen_1_cut:
					if char not in vocab:
						vocab.append(char)

				sen_2_cut, sen_2_slot_name_li, sen_2_slot_name_idx_li, name_start_idx = cut_words(sen_li[1].strip(), slot_name_dict, name_start_idx)

				for idx, slot_name in enumerate(sen_2_slot_name_li):
					slot_name_dict[slot_name] = sen_2_slot_name_idx_li[idx]

				for char in sen_2_cut:
					if char not in vocab:
						vocab.append(char)

				if np.random.randint(1, 11) <= 7: # mt语料
					seed = np.random.randint(1, 11)
					if seed <= 6.0:  # train
						train_mt_fp.write(' '.join(sen_1_cut) + ' | ' + ' '.join(sen_2_cut) + '\n')
						char2vcec_fp.write(' '.join(sen_1_cut) + ' END\n' + ' '.join(sen_2_cut) + ' END\n')
					elif (seed > 6.0 and seed <= 8.0):  # valid
						valid_mt_fp.write(' '.join(sen_1_cut) + ' | ' + ' '.join(sen_2_cut) + '\n')
					else:  # test
						test_mt_fp.write(' '.join(sen_1_cut) + ' | ' + ' '.join(sen_2_cut) + '\n')

				else:  # 无监督语料
					seed = np.random.randint(1, 11)
					source = np.random.randint(0, 2)  # 0/1
					chars_li = None
					if source == 0:
						chars_li = sen_1_cut
					else:
						chars_li = sen_2_cut
					if seed <= 6.0:  # train
						train_fp.write(' '.join(chars_li) + ' | ' + str(source) + '\n')
						char2vcec_fp.write(' '.join(sen_1_cut) + ' END\n' + ' '.join(sen_2_cut) + ' END\n')
					elif (seed > 6.0 and seed <=8.0):  # valid
						valid_fp.write(' '.join(chars_li) + ' | ' + str(source) + '\n')
					else:  # test
						test_fp.write(' '.join(chars_li) + ' | ' + str(source) + '\n')

	for slot_name, slot_name_idx in slot_name_dict.items():
		slot_fp.write(slot_name + '  ' + slot_name_idx + '\n')
		
	vocab_fp.write('PAD' + '\n' + 'GO' + '\n' + 'END' + '\n' + 'NUM_ALP' + '\n' + 'UNK' + '\n')
	for char in vocab:
		vocab_fp.write(char + '\n')

	train_fp.close()
	valid_fp.close()
	test_fp.close()
	train_mt_fp.close()
	valid_mt_fp.close()
	test_mt_fp.close()
	char2vcec_fp.close()
	slot_fp.close()
	vocab_fp.close()


def cut_words(line, slot_name_dict, name_start_idx):
	line = line.strip()
	line = unicode(line.lower())

	chars_li = []
	slots_name_li = re.findall(u'<+[{(\u4E00-\u9FA5|^a-zA-Z0-9\u4e00-\u9fa5|a-z)}]+>', line)

	ret_slots_name_li = []
	ret_slots_name_idx_li = []

	slots_rep_li = re.sub(u'<+[{(\u4E00-\u9FA5|^a-zA-Z0-9\u4e00-\u9fa5|a-z)}]+>', ' slot ', line).strip().split()

	idx = 0
	for words in slots_rep_li:
		if words.strip() == 'slot':
			if slot_name_dict.has_key(slots_name_li[idx].encode('utf-8')):
				slot_name_idx = slot_name_dict[slots_name_li[idx].encode('utf-8')]
				chars_li.append(slot_name_idx.encode('utf-8'))
				idx += 1
			else:
				slot_name_idx = words.encode('utf-8') + '_' + str(name_start_idx)
				chars_li.append(slot_name_idx.encode('utf-8'))
				ret_slots_name_li.append(slots_name_li[idx].encode('utf-8'))
				ret_slots_name_idx_li.append(slot_name_idx.encode('utf-8'))
				name_start_idx += 1
				idx += 1
		else:
			cutted_words = re.findall(u"[\u4E00-\u9FA5]+|[0-9]+|[a-z]+|[^a-zA-Z0-9\u4e00-\u9fa5]", words)
			for word in cutted_words:
				if word.encode('utf-8').isalnum():
					word = word.strip().lower()
					if len(word) > 0:
						chars_li.append('NUM_ALP'.encode("utf-8"))
					continue
				for char in word:
					if len(char.strip()) > 0:
						char = char.encode("utf-8")
						chars_li.append(char)

	return chars_li, ret_slots_name_li, ret_slots_name_idx_li, name_start_idx


if __name__ == "__main__":
	preprocess_speech()
