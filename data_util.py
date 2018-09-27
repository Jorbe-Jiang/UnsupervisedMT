#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import json

import numpy as np
import operator

from config import cfg

np.random.seed(10)

PAD_ID = 0
GO_ID = 1
END_ID = 2
NUM_ALP_ID = 3
UNK_ID = 4


# input_chars | input_source(0/1)
format_train_file = './data/format_train.data'
format_valid_file = './data/format_valid.data'
format_test_file = './data/format_test.data'

# sentence_1 | sentence_2
format_mt_train_file = './data/format_mt_train.data'
format_mt_valid_file = './data/format_mt_valid.data'
format_mt_test_file = './data/format_mt_test.data'

c2v_npy_file = './data/char_vec.npy'

# slot name dict
slot_name_dict_file = './data/slot_name_dict.txt'

vocab_file = './data/vocab.txt'


def word_shuffle(batch_words, batch_lengths, k=3):
    """
    将原来的词序列打乱顺序，使得新的词序列里每个词对应的位置index与之前的位置index之差不大于k，第一个词以及最后一个词不参与打乱
    |new_index(w_i)-w_i| <= k
    """
    new_batch_words = []
    max_words_length = len(batch_words[0])
    for idx, words in enumerate(batch_words):
        ordered_index = []
        selected_index = []
        new_ordered_dict = {}
        for i in range(1, batch_lengths[idx]-1):
            if i in selected_index:
                pass
            else:
                available_idx = list(set(range(i, min(batch_lengths[idx]-1, i + k + 1))).difference(set(ordered_index)))
                new_idx = available_idx[np.random.randint(0, len(available_idx))]
                new_ordered_dict[new_idx] = i
                ordered_index.append(new_idx)

            if new_idx == i or (i in new_ordered_dict):
                continue
            else:
                selected_index.append(i)
                available_idx = list(set(range(i + 1, min(batch_lengths[idx]-1, i + k + 1))).difference(set(selected_index)))
                if len(available_idx) > 0:
                    select_idx = available_idx[np.random.randint(0, len(available_idx))]
                    new_ordered_dict[i] = select_idx
                    if select_idx not in selected_index:
                        selected_index.append(select_idx)

        new_ordered_dict = sorted(new_ordered_dict.items(), key=lambda t: t[0])
        # if len(new_ordered_dict) != (batch_lengths[idx]-2):
        #     print(selected_index)
        #     print(ordered_index)
        #     print(new_ordered_dict)
        #     exit()

        if len(new_ordered_dict) == (batch_lengths[idx]-2):
            new_batch_words.append([words[0]] + [words[word_idx] for (_, word_idx) in new_ordered_dict] + [words[batch_lengths[idx]-1]] + [PAD_ID]*(max_words_length-batch_lengths[idx]))
        else:
            new_batch_words.append(words)
           
    return new_batch_words, batch_lengths


def word_dropout(batch_words, batch_lengths, drop_rate=0.1):
    """
    以概率drop_rate丢弃某个词，第一个词和最后一个词不丢弃
    """
    new_batch_words = []
    new_batch_lengths = []
    new_batch_mask = []
    batch_size = len(batch_words)
    max_length = len(batch_words[0])
    keep = np.random.rand(batch_size, max_length) >= drop_rate
    keep_batch_words = np.asarray(batch_words).astype(np.int32) * keep
    for idx, words in enumerate(keep_batch_words):
        seq_len = batch_lengths[idx]
        new_words = [batch_words[idx][0]] + [w_idx for w_idx in words[1:seq_len-1] if w_idx != 0] + [batch_words[idx][seq_len-1]]
        new_batch_words.append(new_words + [PAD_ID] * (max_length - len(new_words)))
        new_batch_lengths.append(len(new_words))
        new_batch_mask.append([1.0]*len(new_words) + [0.0]*(max_length - len(new_words)))

    # print keep
    # print new_batch_words
    return new_batch_words, new_batch_lengths, new_batch_mask


def add_noise(batch_data, batch_length, k=3, drop_rate=0.1):
    """
    add noise to the encoder input
    """
    batch_data, batch_length = word_shuffle(batch_data, batch_length, k)
    batch_data, batch_length, new_batch_mask = word_dropout(batch_data, batch_length, drop_rate)
    return batch_data, batch_length, new_batch_mask


def load_char2idx_dict():
    return {char.strip():idx for idx, char in enumerate(open(vocab_file, 'r').readlines())}


def load_idx2char_dict():
    return {idx:char.strip() for idx, char in enumerate(open(vocab_file, 'r').readlines())}


def load_slot_name_dict():
    slot_name_dict = {}
    with open(slot_name_dict_file) as fp:
        for idx, line in enumerate(fp.readlines()):
            slot_name, slot_idx = line.strip().split()
            slot_name_dict[slot_idx.strip()] = slot_name.strip()

    return slot_name_dict


# for testing code
def load_mt_testing_data(inputs_chars_length, dec_chars_length, batch_size, mode='train'):
    print('Loading batched data......')
    if mode == 'train':
        open_file = format_mt_train_file

        batched_inputs_li = []
        batched_inputs_mask_li = []
        batched_dec_inputs_li = []
        batched_dec_targets_li = []
        batched_dec_mask_li = []

        n_batches = 0

        one_batch_inputs_li = []
        one_batch_inputs_mask_li = []
        one_batch_dec_inputs_li = []
        one_batch_dec_targets_li = []
        one_batch_dec_mask_li = []
        count = 0

        char2idx_dict = load_char2idx_dict()
        for idx, line in enumerate(open(open_file, 'r').readlines()):
            line = line.strip()

            input_chars, mt_chars = line.split('|')
            input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)
            mt_chars_idx = get_chars_idx_li(mt_chars, char2idx_dict)

            if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2 or len(
                    mt_chars_idx) >= dec_chars_length or len(mt_chars_idx) <= 2:
                continue

            one_batch_inputs_li.append(input_chars_idx + [END_ID] + [PAD_ID] * (inputs_chars_length - len(input_chars_idx) - 1))
            one_batch_inputs_mask_li.append(
                [1.0] * (len(input_chars_idx)+1) + [0.0] * (inputs_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_inputs_li.append(
                [GO_ID] + mt_chars_idx + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
            one_batch_dec_targets_li.append(mt_chars_idx + [END_ID] + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
            one_batch_dec_mask_li.append([1.0] * (len(mt_chars_idx)+1) + [0.0] * (dec_chars_length - len(mt_chars_idx) - 1))

            count += 1

            if count % batch_size == 0:
                # [batch_size, [inputs_chars_length]]
                batched_inputs_li.append(one_batch_inputs_li)
                batched_inputs_mask_li.append(one_batch_inputs_mask_li)

                # [dec_chars_length, [batch_size]]
                batched_dec_targets_li.append(
                    [[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_mask_li.append(
                    [[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_inputs_li.append(
                    [[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

                n_batches += 1
                count = 0
                one_batch_inputs_li = []
                one_batch_inputs_mask_li = []
                one_batch_dec_inputs_li = []
                one_batch_dec_targets_li = []
                one_batch_dec_mask_li = []

            if len(batched_inputs_li) == 10:
                print('End...')
                return (batched_inputs_li, batched_inputs_mask_li, batched_dec_inputs_li, batched_dec_targets_li, batched_dec_mask_li, n_batches)

    elif mode == 'valid':
        open_file = format_mt_valid_file

        batched_inputs_li = []
        batched_inputs_mask_li = []
        batched_dec_inputs_li = []
        batched_dec_targets_li = []
        batched_dec_mask_li = []

        n_batches = 0

        one_batch_inputs_li = []
        one_batch_inputs_mask_li = []
        one_batch_dec_inputs_li = []
        one_batch_dec_targets_li = []
        one_batch_dec_mask_li = []
        count = 0

        char2idx_dict = load_char2idx_dict()
        for idx, line in enumerate(open(open_file, 'r').readlines()):
            line = line.strip()

            input_chars, mt_chars = line.split('|')
            input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)
            mt_chars_idx = get_chars_idx_li(mt_chars, char2idx_dict)

            if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2 or len(
                    mt_chars_idx) >= dec_chars_length or len(mt_chars_idx) <= 2:
                continue

            one_batch_inputs_li.append(
                input_chars_idx + [END_ID] + [PAD_ID] * (inputs_chars_length - len(input_chars_idx) - 1))
            one_batch_inputs_mask_li.append(
                [1.0] * (len(input_chars_idx) + 1) + [0.0] * (inputs_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_inputs_li.append(
                [GO_ID] + mt_chars_idx + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
            one_batch_dec_targets_li.append(
                mt_chars_idx + [END_ID] + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
            one_batch_dec_mask_li.append(
                [1.0] * (len(mt_chars_idx) + 1) + [0.0] * (dec_chars_length - len(mt_chars_idx) - 1))

            count += 1

            if count % batch_size == 0:
                # [batch_size, [inputs_chars_length]]
                batched_inputs_li.append(one_batch_inputs_li)
                batched_inputs_mask_li.append(one_batch_inputs_mask_li)

                # [dec_chars_length, [batch_size]]
                batched_dec_targets_li.append(
                    [[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_mask_li.append(
                    [[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_inputs_li.append(
                    [[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

                n_batches += 1
                count = 0
                one_batch_inputs_li = []
                one_batch_inputs_mask_li = []
                one_batch_dec_inputs_li = []
                one_batch_dec_targets_li = []
                one_batch_dec_mask_li = []

            if len(batched_inputs_li) == 10:
                print('End...')
                return (batched_inputs_li, batched_inputs_mask_li, batched_dec_inputs_li, batched_dec_targets_li,
                        batched_dec_mask_li, n_batches)
    else:
        pass


# for testing code
def load_testing_data(inputs_chars_length, dec_chars_length, batch_size, mode='train'):
    print('Loading batched data......')
    if mode == 'train':
        open_file = format_train_file

        batched_inputs_noise_chars_li = []
        batched_sources_li = []
        batched_mt_sources_li = []
        batched_inputs_mask_li = []
        batched_inputs_seq_len_li = []
        batched_dec_inputs_li = []
        batched_dec_targets_li = []
        batched_dec_mask_li = []

        n_batches = 0

        one_batch_inputs_noise_chars_li = []
        one_batch_sources_li = []
        one_batch_mt_sources_li = []
        one_batch_inputs_mask_li = []
        one_batch_inputs_seq_len_li = []
        one_batch_dec_inputs_li = []
        one_batch_dec_targets_li = []
        one_batch_dec_mask_li = []
        count = 0

        char2idx_dict = load_char2idx_dict()
        for idx, line in enumerate(open(open_file, 'r').readlines()):
            line = line.strip()

            input_chars, input_source = line.split('|')
            input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)

            if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2:
                continue

            one_batch_sources_li.append(int(input_source))
            one_batch_mt_sources_li.append(abs(int(input_source) - 1))
            one_batch_inputs_seq_len_li.append(len(input_chars_idx) + 1)
            one_batch_dec_inputs_li.append(
                [GO_ID] + input_chars_idx + [PAD_ID] * (dec_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_targets_li.append(
                input_chars_idx + [END_ID] + [PAD_ID] * (dec_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_mask_li.append(
                [1.0] * (len(input_chars_idx) + 1) + [0.0] * (dec_chars_length - len(input_chars_idx) - 1))

            count += 1

            if count % batch_size == 0:
                # [batch_size]
                batched_sources_li.append(one_batch_sources_li)
                batched_mt_sources_li.append(one_batch_mt_sources_li)

                one_batch_inputs_noise_chars_li, one_batch_inputs_seq_len_li, one_batch_inputs_mask_li = add_noise(
                    one_batch_dec_targets_li, one_batch_inputs_seq_len_li)
                # [batch_size, [inputs_chars_length]]
                batched_inputs_noise_chars_li.append(one_batch_inputs_noise_chars_li)
                batched_inputs_seq_len_li.append(one_batch_inputs_seq_len_li)
                batched_inputs_mask_li.append(one_batch_inputs_mask_li)

                # [dec_chars_length, [batch_size]]
                batched_dec_targets_li.append(
                    [[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_mask_li.append(
                    [[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_inputs_li.append(
                    [[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

                n_batches += 1
                count = 0
                one_batch_inputs_noise_chars_li = []
                one_batch_sources_li = []
                one_batch_mt_sources_li = []
                one_batch_inputs_mask_li = []
                one_batch_inputs_seq_len_li = []
                one_batch_dec_inputs_li = []
                one_batch_dec_targets_li = []
                one_batch_dec_mask_li = []

            if len(batched_inputs_noise_chars_li) == 10:
                print('END...')
                return (batched_inputs_noise_chars_li, batched_sources_li, batched_mt_sources_li, batched_inputs_mask_li, batched_inputs_seq_len_li, batched_dec_inputs_li, batched_dec_targets_li, batched_dec_mask_li, n_batches)

    elif mode == 'valid':
        open_file = format_valid_file

        batched_inputs_noise_chars_li = []
        batched_sources_li = []
        batched_mt_sources_li = []
        batched_inputs_mask_li = []
        batched_inputs_seq_len_li = []
        batched_dec_inputs_li = []
        batched_dec_targets_li = []
        batched_dec_mask_li = []

        n_batches = 0

        one_batch_inputs_noise_chars_li = []
        one_batch_sources_li = []
        one_batch_mt_sources_li = []
        one_batch_inputs_mask_li = []
        one_batch_inputs_seq_len_li = []
        one_batch_dec_inputs_li = []
        one_batch_dec_targets_li = []
        one_batch_dec_mask_li = []
        count = 0

        char2idx_dict = load_char2idx_dict()
        for idx, line in enumerate(open(open_file, 'r').readlines()):
            line = line.strip()

            input_chars, input_source = line.split('|')
            input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)

            if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2:
                continue

            one_batch_sources_li.append(int(input_source))
            one_batch_mt_sources_li.append(abs(int(input_source) - 1))
            one_batch_inputs_seq_len_li.append(len(input_chars_idx) + 1)
            one_batch_dec_inputs_li.append(
                [GO_ID] + input_chars_idx + [PAD_ID] * (dec_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_targets_li.append(
                input_chars_idx + [END_ID] + [PAD_ID] * (dec_chars_length - len(input_chars_idx) - 1))
            one_batch_dec_mask_li.append(
                [1.0] * (len(input_chars_idx) + 1) + [0.0] * (dec_chars_length - len(input_chars_idx) - 1))

            count += 1

            if count % batch_size == 0:
                # [batch_size]
                batched_sources_li.append(one_batch_sources_li)
                batched_mt_sources_li.append(one_batch_mt_sources_li)

                one_batch_inputs_noise_chars_li, one_batch_inputs_seq_len_li, one_batch_inputs_mask_li = add_noise(
                    one_batch_dec_targets_li, one_batch_inputs_seq_len_li)
                # [batch_size, [inputs_chars_length]]
                batched_inputs_noise_chars_li.append(one_batch_inputs_noise_chars_li)
                batched_inputs_seq_len_li.append(one_batch_inputs_seq_len_li)
                batched_inputs_mask_li.append(one_batch_inputs_mask_li)

                # [dec_chars_length, [batch_size]]
                batched_dec_targets_li.append(
                    [[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_mask_li.append(
                    [[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
                batched_dec_inputs_li.append(
                    [[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

                n_batches += 1
                count = 0
                one_batch_inputs_noise_chars_li = []
                one_batch_sources_li = []
                one_batch_mt_sources_li = []
                one_batch_inputs_mask_li = []
                one_batch_inputs_seq_len_li = []
                one_batch_dec_inputs_li = []
                one_batch_dec_targets_li = []
                one_batch_dec_mask_li = []

            if len(batched_inputs_noise_chars_li) == 10:
                print('END...')
                return (
                batched_inputs_noise_chars_li, batched_sources_li, batched_mt_sources_li, batched_inputs_mask_li,
                batched_inputs_seq_len_li, batched_dec_inputs_li, batched_dec_targets_li, batched_dec_mask_li,
                n_batches)
    else:
        pass


def load_mt_data(inputs_chars_length, dec_chars_length, batch_size, mode='train'):
    print('Loading batched data......')
    if mode == 'train':
        train_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_mt_batched_train.npy'
        if os.path.exists(train_file):
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                train_file)
        else:
            create_mt_dataset(inputs_chars_length, dec_chars_length, batch_size, train_file, 'train')
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                train_file)

        print('End...')
        return (batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    elif mode == 'valid':
        valid_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_mt_batched_valid.npy'
        if os.path.exists(valid_file):
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                valid_file)
        else:
            create_mt_dataset(inputs_chars_length, dec_chars_length, batch_size, valid_file, 'valid')
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                valid_file)

        print('End...')
        return (batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    else:
        test_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_mt_batched_test.npy'
        if os.path.exists(test_file):
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                test_file)
        else:
            create_mt_dataset(inputs_chars_length, dec_chars_length, batch_size, test_file, 'test')
            batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                test_file)

        print('End...')
        return (batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)


def load_data(inputs_chars_length, dec_chars_length, batch_size, mode='train'):
    print('Loading batched data......')
    if mode == 'train':
        train_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_batched_train.npy'
        if os.path.exists(train_file):
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                train_file)
        else:
            create_dataset(inputs_chars_length, dec_chars_length, batch_size, train_file, 'train')
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                train_file)

        print('End...')
        return (batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    elif mode == 'valid':
        valid_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_batched_valid.npy'
        if os.path.exists(valid_file):
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                valid_file)
        else:
            create_dataset(inputs_chars_length, dec_chars_length, batch_size, valid_file, 'valid')
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                valid_file)

        print('End...')
        return (batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    else:
        test_file = cfg.data_path + str(batch_size) + '_' + str(inputs_chars_length) + '_' + str(dec_chars_length) + '_batched_test.npy'
        if os.path.exists(test_file):
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                test_file)
        else:
            create_dataset(inputs_chars_length, dec_chars_length, batch_size, test_file, 'test')
            batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = np.load(
                test_file)

        print('End...')
        return (batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)


def get_chars_idx_li(input_chars, char2idx_dict):
    input_chars_idx = []
    for c in input_chars.split():
        try:
            char_idx = char2idx_dict[c]
            input_chars_idx.append(char_idx)
        except:
            if c.encode('utf-8').isalnum():
                char_idx = char2idx_dict['NUM_ALP']
            else:
                char_idx = char2idx_dict['UNK']
            input_chars_idx.append(char_idx)

    return input_chars_idx


def create_mt_dataset(inputs_chars_length, dec_chars_length, batch_size, output_file, mode='train'):
    if mode == 'train':
        open_file = format_mt_train_file

    elif mode == 'valid':
        open_file = format_mt_valid_file

    else:
        open_file = format_mt_test_file

    batched_inputs_li = []
    batched_inputs_mask_li = []
    batched_dec_inputs_li = []
    batched_dec_targets_li = []
    batched_dec_mask_li = []

    n_batches = 0

    one_batch_inputs_li = []
    one_batch_inputs_mask_li = []
    one_batch_dec_inputs_li = []
    one_batch_dec_targets_li = []
    one_batch_dec_mask_li = []
    count = 0

    char2idx_dict = load_char2idx_dict()
    for idx, line in enumerate(open(open_file, 'r').readlines()):
        line = line.strip()

        input_chars, mt_chars = line.split('|')
        input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)
        mt_chars_idx = get_chars_idx_li(mt_chars, char2idx_dict)

        if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2 or len(mt_chars_idx) >= dec_chars_length or len(mt_chars_idx) <= 2:
            continue

        one_batch_inputs_li.append(input_chars_idx + [END_ID] + [PAD_ID] * (inputs_chars_length - len(input_chars_idx) - 1))
        one_batch_inputs_mask_li.append([1.0] * (len(input_chars_idx) + 1) + [0.0] * (inputs_chars_length - len(input_chars_idx) - 1))
        one_batch_dec_inputs_li.append([GO_ID] + mt_chars_idx + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
        one_batch_dec_targets_li.append(mt_chars_idx + [END_ID] + [PAD_ID] * (dec_chars_length - len(mt_chars_idx) - 1))
        one_batch_dec_mask_li.append([1.0] * (len(mt_chars_idx) + 1) + [0.0] * (dec_chars_length - len(mt_chars_idx) - 1))

        count += 1

        if count % batch_size == 0:
            # [batch_size, [inputs_chars_length]]
            batched_inputs_li.append(one_batch_inputs_li)
            batched_inputs_mask_li.append(one_batch_inputs_mask_li)

            # [dec_chars_length, [batch_size]]
            batched_dec_targets_li.append([[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
            batched_dec_mask_li.append([[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
            batched_dec_inputs_li.append([[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

            n_batches += 1
            count = 0
            one_batch_inputs_li = []
            one_batch_inputs_mask_li = []
            one_batch_dec_inputs_li = []
            one_batch_dec_targets_li = []
            one_batch_dec_mask_li = []

    one_batch_inputs_li = []
    one_batch_inputs_mask_li = []
    one_batch_dec_inputs_li = []
    one_batch_dec_targets_li = []
    one_batch_dec_mask_li = []

    np.save(output_file, (batched_inputs_li, batched_inputs_mask_li, batched_dec_inputs_li, batched_dec_targets_li, batched_dec_mask_li, n_batches))


def create_dataset(inputs_chars_length, dec_chars_length, batch_size, output_file, mode='train'):
    if mode == 'train':
        open_file = format_train_file

    elif mode == 'valid':
        open_file = format_valid_file

    else:
        open_file = format_test_file

    batched_inputs_noise_chars_li = []
    batched_sources_li = []
    batched_mt_sources_li = []
    batched_inputs_mask_li = []
    batched_inputs_seq_len_li = []
    batched_dec_inputs_li = []
    batched_dec_targets_li = []
    batched_dec_mask_li = []

    n_batches = 0

    one_batch_inputs_noise_chars_li = []
    one_batch_sources_li = []
    one_batch_mt_sources_li = []
    one_batch_inputs_mask_li = []
    one_batch_inputs_seq_len_li = []
    one_batch_dec_inputs_li = []
    one_batch_dec_targets_li = []
    one_batch_dec_mask_li = []
    count = 0

    char2idx_dict = load_char2idx_dict()
    for idx, line in enumerate(open(open_file, 'r').readlines()):
        line = line.strip()

        input_chars, input_source = line.split('|')
        input_chars_idx = get_chars_idx_li(input_chars, char2idx_dict)

        if len(input_chars_idx) >= inputs_chars_length or len(input_chars_idx) <= 2:
            continue

        one_batch_sources_li.append(int(input_source))
        one_batch_mt_sources_li.append(abs(int(input_source)-1))
        one_batch_inputs_seq_len_li.append(len(input_chars_idx)+1)
        one_batch_dec_inputs_li.append([GO_ID] + input_chars_idx + [PAD_ID]*(dec_chars_length - len(input_chars_idx) - 1))
        one_batch_dec_targets_li.append(input_chars_idx + [END_ID] + [PAD_ID]*(dec_chars_length - len(input_chars_idx) - 1))
        one_batch_dec_mask_li.append([1.0]*(len(input_chars_idx)+1) + [0.0]*(dec_chars_length - len(input_chars_idx) - 1))

        count += 1

        if count % batch_size == 0:
            # [batch_size]
            batched_sources_li.append(one_batch_sources_li)
            batched_mt_sources_li.append(one_batch_mt_sources_li)

            one_batch_inputs_noise_chars_li, one_batch_inputs_seq_len_li, one_batch_inputs_mask_li = add_noise(one_batch_dec_targets_li, one_batch_inputs_seq_len_li)
            # [batch_size, [inputs_chars_length]]
            batched_inputs_noise_chars_li.append(one_batch_inputs_noise_chars_li)
            batched_inputs_seq_len_li.append(one_batch_inputs_seq_len_li)
            batched_inputs_mask_li.append(one_batch_inputs_mask_li)

            # [dec_chars_length, [batch_size]]
            batched_dec_targets_li.append([[one_batch_dec_targets_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
            batched_dec_mask_li.append([[one_batch_dec_mask_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])
            batched_dec_inputs_li.append([[one_batch_dec_inputs_li[j][i] for j in range(batch_size)] for i in range(dec_chars_length)])

            n_batches += 1
            count = 0
            one_batch_inputs_noise_chars_li = []
            one_batch_sources_li = []
            one_batch_mt_sources_li = []
            one_batch_inputs_mask_li = []
            one_batch_inputs_seq_len_li = []
            one_batch_dec_inputs_li = []
            one_batch_dec_targets_li = []
            one_batch_dec_mask_li = []

    one_batch_inputs_noise_chars_li = []
    one_batch_sources_li = []
    one_batch_mt_sources_li = []
    one_batch_inputs_mask_li = []
    one_batch_inputs_seq_len_li = []
    one_batch_dec_inputs_li = []
    one_batch_dec_targets_li = []
    one_batch_dec_mask_li = []

    np.save(output_file, (batched_inputs_noise_chars_li, batched_sources_li, batched_mt_sources_li, batched_inputs_mask_li, batched_inputs_seq_len_li, batched_dec_inputs_li, batched_dec_targets_li, batched_dec_mask_li, n_batches))


########################################################################################################################
if __name__ == '__main__':
    pass
