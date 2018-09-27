#!/usr/bin/python
# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import sys

import data_util
from model import Model
from config import cfg

tf.set_random_seed(10)
np.random.seed(10)

PAD_ID = 0
GO_ID = 1
END_ID = 2
NUM_ALP_ID = 3
UNK_ID = 4


def try_make_model_dir():
    dir = './models'
    if os.path.exists(dir) and os.path.isdir(dir):
        return
    os.makedirs(dir)


def get_model_path():
    return './models/model.ckpt'


def get_model_dir():
    return './models/'


char2idx_dict = data_util.load_char2idx_dict()
idx2char_dict = data_util.load_idx2char_dict()
slot_name_dict = data_util.load_slot_name_dict()


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
            # print(selected_index)
            # print(ordered_index)
            # print(new_ordered_dict)
            # exit()

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

    return new_batch_words, new_batch_lengths, new_batch_mask


def add_noise(batch_data, batch_length, k=3, drop_rate=0.1):
    """
    add noise to the encoder input
    """
    batch_data, batch_length = word_shuffle(batch_data, batch_length, k)
    batch_data, batch_length, new_batch_mask = word_dropout(batch_data, batch_length, drop_rate)
    return batch_data, batch_length, new_batch_mask


def print_mt_eval_results(batched_inputs, batched_targets, batched_preds, step):
    """
    print MT translated results
    batched_inputs: list of batch data, element shape is [[seq_len] of batch_size]
    batched_targets: list of batch data, element shape is [[batch_size] of seq_len]
    batched_preds: translated data
    """
    fp = open('mt_eval_results_{}.txt'.format(step), 'w')

    batch_size = len(batched_inputs[0])
    for idx, batch_inputs in enumerate(batched_inputs):
        # [batch_size, seq_len]
        batch_targets = np.transpose(np.asarray(batched_targets[idx], dtype=np.int32))
        batch_preds = np.transpose(np.asarray(batched_preds[idx], dtype=np.int32))

        for i in range(batch_size):
            targets = batch_targets[i].tolist()
            preds = batch_preds[i].tolist()

            inputs = batch_inputs[i][:batch_inputs[i].index(END_ID)+1]
            targets = targets[:targets.index(END_ID) + 1]
            try:
                preds = preds[:preds.index(END_ID) + 1]
            except:
                preds = preds[:]

            inputs = [idx2char_dict[char_idx] for char_idx in inputs]
            targets = [idx2char_dict[char_idx] for char_idx in targets]
            preds = [idx2char_dict[char_idx] for char_idx in preds]

            inputs = [slot_name_dict[char] if 'slot_' in char else char for char in inputs]
            targets = [slot_name_dict[char] if 'slot_' in char else char for char in targets]
            preds = [slot_name_dict[char] if 'slot_' in char else char for char in preds]
            fp.write(' '.join(inputs) + ' | ' + ' '.join(preds) + '\n')

    fp.close()


def print_eval_resutls(batched_targets, batched_preds, step):
    """
    print model results
    batched_targets: list of batch data, element shape is [[batch_size] of seq_len]
    batched_preds: translated data
    """
    fp = open('eval_results_{}.txt'.format(step), 'w')

    batch_size = len(batched_targets[0][0])
    for idx, batch_targets in enumerate(batched_targets):
        # [batch_size, seq_len]
        batch_targets = np.transpose(np.asarray(batch_targets, dtype=np.int32))
        batch_preds = np.transpose(np.asarray(batched_preds[idx], dtype=np.int32))

        for i in range(batch_size):
            targets = batch_targets[i].tolist()
            preds = batch_preds[i].tolist()

            targets = targets[:targets.index(END_ID)+1]
            try:
                preds = preds[:preds.index(END_ID)+1]
            except:
                preds = preds[:]

            targets = [idx2char_dict[char_idx] for char_idx in targets]
            preds = [idx2char_dict[char_idx] for char_idx in preds]

            targets = [slot_name_dict[char] if 'slot_' in char else char for char in targets]
            preds = [slot_name_dict[char] if 'slot_' in char else char for char in preds]
            fp.write(' '.join(targets) + ' | ' + ' '.join(preds) + '\n')

    fp.close()


def evaluation_mt(model, sess, valid_data, best_loss, step):
    batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = valid_data

    loss_li = []
    predictions_li = []
    cfg.mode = 'valid'
    for i in range(n_batches):
        inputs_mt_chars = np.asarray(batched_inputs[i], dtype=np.int32)
        inputs_mt_mask = np.asarray(batched_inputs_mask[i], dtype=np.float32)
        inputs_seq_length = np.asarray(np.sum(inputs_mt_mask, axis=1), dtype=np.int32)

        dec_inputs = batched_dec_inputs[i]
        dec_targets = batched_dec_targets[i]
        dec_mask = batched_dec_mask[i]

        feed_dict = {}
        for t in range(cfg.dec_chars_length):
            feed_dict[model.dec_inputs[t]] = [GO_ID]*cfg.batch_size
            feed_dict[model.dec_targets[t]] = dec_targets[t]
            feed_dict[model.dec_mask[t]] = dec_mask[t]

        feed_dict[model.inputs_mt_noise_chars] = inputs_mt_chars
        feed_dict[model.inputs_mt_mask] = inputs_mt_mask
        feed_dict[model.inputs_mt_seq_length] = inputs_seq_length

        feed_dict[model.keep_rate] = 1.0
        feed_dict[model.mode] = 'valid'

        (loss, valid_predictions) = sess.run((model.mt_loss, model.valid_predictions), feed_dict=feed_dict)

        loss_li.append(loss)
        predictions_li.append(valid_predictions)

    if np.mean(loss_li) < best_loss and step >= 20000:
        print_mt_eval_results(batched_inputs, batched_dec_targets, predictions_li, step)

    return np.mean(loss_li)


def evaluation(model, sess, valid_data, best_loss, step):
    batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = valid_data

    loss_li = []
    predictions_li = []
    cfg.mode = 'valid'
    for i in range(n_batches):
        dec_inputs = batched_dec_inputs[i]
        dec_targets = batched_dec_targets[i]
        dec_mask = batched_dec_mask[i]

        # 生成翻译句子
        inputs_mt_chars = np.reshape(np.asarray(dec_targets, dtype=np.int32), (cfg.dec_chars_length, cfg.batch_size))
        inputs_mt_mask = np.reshape(np.asarray(dec_mask, dtype=np.float32), (cfg.dec_chars_length, cfg.batch_size))
        # [batch_size, seq_length]
        inputs_mt_chars = np.transpose(inputs_mt_chars)
        inputs_mt_mask = np.transpose(inputs_mt_mask)

        inputs_mt_seq_length = np.asarray(np.sum(inputs_mt_mask, axis=1), dtype=np.int32)

        mt_feed_dict = {}
        for t in range(cfg.dec_chars_length):
            mt_feed_dict[model.dec_inputs[t]] = [GO_ID] * cfg.batch_size

        mt_feed_dict[model.inputs_mt_noise_chars] = inputs_mt_chars
        mt_feed_dict[model.inputs_mt_mask] = inputs_mt_mask
        mt_feed_dict[model.inputs_mt_seq_length] = inputs_mt_seq_length
        mt_feed_dict[model.keep_rate] = 1.0

        # [seq_length, batch_size]
        mt_predictions = sess.run(model.valid_predictions, feed_dict=mt_feed_dict)

        predictions_li.append(mt_predictions)

        mt_predictions = np.transpose(mt_predictions).tolist()
        mt_pred_seq_length = []
        for prediction in mt_predictions:
            end_idx = prediction.index(END_ID)
            mt_pred_seq_length.append(end_idx + 1)

        inputs_mt_noise_chars, inputs_mt_seq_length, inputs_mt_mask = add_noise(mt_predictions, mt_pred_seq_length)

        inputs_noise_chars = np.asarray(batched_inputs_noise_chars[i], dtype=np.int32)
        inputs_mt_noise_chars = np.asarray(inputs_mt_noise_chars, dtype=np.int32)
        inputs_sources = np.asarray(batched_sources[i], dtype=np.int32)
        inputs_mask = np.asarray(batched_inputs_mask[i], dtype=np.float32)
        inputs_mt_mask = np.asarray(inputs_mt_mask, dtype=np.float32)
        inputs_seq_length = np.asarray(batched_inputs_seq_length[i], dtype=np.int32)
        inputs_mt_seq_length = np.asarray(inputs_mt_seq_length, dtype=np.int32)

        inputs_mt_sources = np.asarray(batched_mt_sources[i], dtype=np.int32)

        feed_dict = {}
        for t in range(cfg.dec_chars_length):
            feed_dict[model.dec_inputs[t]] = [GO_ID]*cfg.batch_size
            feed_dict[model.dec_targets[t]] = dec_targets[t]
            feed_dict[model.dec_mask[t]] = dec_mask[t]

        feed_dict[model.inputs_noise_chars] = inputs_noise_chars
        feed_dict[model.inputs_sources] = inputs_sources
        feed_dict[model.inputs_mask] = inputs_mask
        feed_dict[model.inputs_mt_mask] = inputs_mt_mask

        feed_dict[model.inputs_mt_sources] = inputs_mt_sources
        feed_dict[model.inputs_mt_noise_chars] = inputs_mt_noise_chars

        feed_dict[model.inputs_seq_length] = inputs_seq_length
        feed_dict[model.inputs_mt_seq_length] = inputs_mt_seq_length
        feed_dict[model.keep_rate] = 1.0
        feed_dict[model.mode] = 'valid'

        (loss, valid_predictions) = sess.run((model.total_loss, model.valid_predictions), feed_dict=feed_dict)

        loss_li.append(loss)

    if np.mean(loss_li) < best_loss and step >= 20000:
        print_eval_resutls(batched_dec_targets, predictions_li, step)

    return np.mean(loss_li)


def train_mt():
    print('start train MT model')
    batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = data_util.load_mt_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'train')
    #valid_mt_data: (batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    valid_mt_data = data_util.load_mt_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'valid')

    # # for testing code
    # batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = data_util.load_mt_testing_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'train')
    # # valid_mt_data: (batched_inputs, batched_inputs_mask, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    # valid_mt_data = data_util.load_mt_testing_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'valid')

    with tf.device(cfg.device):
        m = Model(cfg.char_size, cfg.char_vec_dim)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    curr_mean_loss = 0.0
    prev_mean_loss = 0.0
    loss_add_counts = 0
    best_loss = 100000000.0

    char_embedding = np.load(cfg.data_path+"char_vec.npy")

    model_path = './mt_models/'
    with tf.Session(config=config) as sess:
        m.summary_writer.add_graph(sess.graph)
        try_make_model_dir()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('Loading pre-trained model from: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            # reset learning rate
            sess.run(m.mt_lr.assign(cfg.mt_lr))
        else:
            tf.global_variables_initializer().run()

            with tf.variable_scope("char_embedding_layer", reuse=True):
                char_emb = tf.get_variable("char_embedding")

            with tf.variable_scope("Decoder/embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
                em_out = tf.get_variable("embedding")

            # Disable training for embeddings
            variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            variables.remove(em_out)

            # Initialize input and output embeddings
            sess.run(char_emb.assign(char_embedding))
            sess.run(em_out.assign(char_embedding))

        for epoch in range(cfg.num_epochs):
            loss_li = []
            shuffle_idx_li = np.random.permutation(n_batches)

            for i in range(n_batches):
                inputs_mt_chars = np.asarray(batched_inputs[shuffle_idx_li[i]], dtype=np.int32)
                inputs_mt_mask = np.asarray(batched_inputs_mask[shuffle_idx_li[i]], dtype=np.float32)
                inputs_seq_length = np.asarray(np.sum(inputs_mt_mask, axis=1), dtype=np.int32)

                dec_inputs = batched_dec_inputs[shuffle_idx_li[i]]
                dec_targets = batched_dec_targets[shuffle_idx_li[i]]
                dec_mask = batched_dec_mask[shuffle_idx_li[i]]

                feed_dict = {}
                for t in range(cfg.dec_chars_length):
                    feed_dict[m.dec_inputs[t]] = dec_inputs[t]
                    feed_dict[m.dec_targets[t]] = dec_targets[t]
                    feed_dict[m.dec_mask[t]] = dec_mask[t]

                feed_dict[m.inputs_mt_noise_chars] = inputs_mt_chars
                feed_dict[m.inputs_mt_mask] = inputs_mt_mask
                feed_dict[m.inputs_mt_seq_length] = inputs_seq_length

                feed_dict[m.keep_rate] = cfg.keep_rate
                feed_dict[m.mode] = cfg.mode

                cfg.mode = 'train'

                (mt_lr, mt_loss, predictions, _) = sess.run((m.mt_lr, m.mt_loss, m.predictions, m.mt_train_op), feed_dict=feed_dict)

                loss_li.append(mt_loss)

                step = i + epoch*n_batches

                if step % 10 == 1:
                    print('Epoch: %s step: %s mt_lr: %s mt_loss: %s' % (
                        epoch, step, mt_lr, mt_loss))

                if (step > 0 and step <= n_batches and step % int(n_batches/2) == 0) or (step % int(n_batches/4) == 0 and step > n_batches):
                    eval_loss = evaluation_mt(m, sess, valid_mt_data, best_loss, step)
                    curr_mean_loss = eval_loss
                    # decrease learning rate if no improvement and early stopping
                    if curr_mean_loss >= prev_mean_loss and prev_mean_loss != 0.0 and step >= cfg.start_decay_at:
                        loss_add_counts += 1
                        # 如果连续三次loss都increase，则early stopping
                        if loss_add_counts >= 3:
                            print('Early Stopping!!!')
                            exit()
                        if mt_lr > 0.00005:
                            sess.run(m.mt_lr_decay_op)
                    elif curr_mean_loss < prev_mean_loss and prev_mean_loss != 0.0:
                        loss_add_counts = 0

                    prev_mean_loss = curr_mean_loss

                    if curr_mean_loss < best_loss:
                        best_loss = curr_mean_loss
                        saved_model_file = '{}_{}_{:.3f}.ckpt'.format(epoch, step, best_loss)
                        saved_model_path = os.path.join(model_path, saved_model_file)
                        print('model saved in file : %s' % saved_model_path)
                        saver.save(sess, saved_model_path)

                if step > n_batches:
                    cfg.keep_rate = np.minimum(cfg.keep_rate * 1.01, 1.0)


def train():
    print('start train')
    batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = data_util.load_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'train')
    # valid_data: (batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    valid_data = data_util.load_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'valid')

    # # for testing code
    # batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches = data_util.load_testing_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'train')
    # # valid_data: (batched_inputs_noise_chars, batched_sources, batched_mt_sources, batched_inputs_mask, batched_inputs_seq_length, batched_dec_inputs, batched_dec_targets, batched_dec_mask, n_batches)
    # valid_data = data_util.load_testing_data(cfg.inputs_chars_length, cfg.dec_chars_length, cfg.batch_size, 'valid')

    with tf.device(cfg.device):
        m = Model(cfg.char_size, cfg.char_vec_dim)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    merged_sum = tf.summary.merge_all()

    curr_mean_loss = 0.0
    prev_mean_loss = 0.0
    loss_add_counts = 0
    best_loss = 100000000.0

    best_dis_loss = 1000000000.0

    char_embedding = np.load(cfg.data_path+"char_vec.npy")

    model_path = get_model_dir()
    mt_model_path = './mt_models/'
    with tf.Session(config=config) as sess:
        m.summary_writer.add_graph(sess.graph)
        try_make_model_dir()
        ckpt = tf.train.get_checkpoint_state(mt_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('Loading pre-trained model from: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            # reset learning rate
            sess.run(m.lr.assign(cfg.lr))
        else:
            print('No pre-trained MT model...')
            exit()

        for epoch in range(cfg.num_epochs):
            loss_li = []
            dis_loss_li = []
            shuffle_idx_li = np.random.permutation(n_batches)

            for i in range(n_batches):
                dec_inputs = batched_dec_inputs[shuffle_idx_li[i]]
                dec_targets = batched_dec_targets[shuffle_idx_li[i]]
                dec_mask = batched_dec_mask[shuffle_idx_li[i]]

                # 生成翻译句子
                inputs_mt_chars = np.reshape(np.asarray(dec_targets, dtype=np.int32), (cfg.dec_chars_length, cfg.batch_size))
                inputs_mt_mask = np.reshape(np.asarray(dec_mask, dtype=np.float32), (cfg.dec_chars_length, cfg.batch_size))
                # [batch_size, seq_length]
                inputs_mt_chars = np.transpose(inputs_mt_chars)
                inputs_mt_mask = np.transpose(inputs_mt_mask)

                inputs_mt_seq_length = np.asarray(np.sum(inputs_mt_mask, axis=1), dtype=np.int32)

                mt_feed_dict = {}
                for t in range(cfg.dec_chars_length):
                    mt_feed_dict[m.dec_inputs[t]] = [GO_ID]*cfg.batch_size

                mt_feed_dict[m.inputs_mt_noise_chars] = inputs_mt_chars
                mt_feed_dict[m.inputs_mt_mask] = inputs_mt_mask
                mt_feed_dict[m.inputs_mt_seq_length] = inputs_mt_seq_length
                mt_feed_dict[m.keep_rate] = 1.0

                # [seq_length, batch_size]
                mt_predictions = sess.run(m.valid_predictions, feed_dict=mt_feed_dict)

                mt_predictions = np.transpose(mt_predictions).tolist()
                mt_pred_seq_length = []
                for prediction in mt_predictions:
                    end_idx = prediction.index(END_ID)
                    mt_pred_seq_length.append(end_idx+1)

                inputs_mt_noise_chars, inputs_mt_seq_length, inputs_mt_mask = add_noise(mt_predictions, mt_pred_seq_length)

                inputs_noise_chars = np.asarray(batched_inputs_noise_chars[shuffle_idx_li[i]], dtype=np.int32)
                inputs_mt_noise_chars = np.asarray(inputs_mt_noise_chars, dtype=np.int32)
                inputs_sources = np.asarray(batched_sources[shuffle_idx_li[i]], dtype=np.int32)
                inputs_mask = np.asarray(batched_inputs_mask[shuffle_idx_li[i]], dtype=np.float32)
                inputs_mt_mask = np.asarray(inputs_mt_mask, dtype=np.float32)
                inputs_seq_length = np.asarray(batched_inputs_seq_length[shuffle_idx_li[i]], dtype=np.int32)
                inputs_mt_seq_length = np.asarray(inputs_mt_seq_length, dtype=np.int32)

                inputs_mt_sources = np.asarray(batched_mt_sources[shuffle_idx_li[i]], dtype=np.int32)

                feed_dict = {}
                dis_feed_dict = {}
                for t in range(cfg.dec_chars_length):
                    feed_dict[m.dec_inputs[t]] = dec_inputs[t]
                    feed_dict[m.dec_targets[t]] = dec_targets[t]
                    feed_dict[m.dec_mask[t]] = dec_mask[t]

                feed_dict[m.inputs_noise_chars] = inputs_noise_chars
                feed_dict[m.inputs_sources] = inputs_sources
                feed_dict[m.inputs_mask] = inputs_mask
                feed_dict[m.inputs_mt_mask] = inputs_mt_mask

                feed_dict[m.inputs_mt_sources] = inputs_mt_sources
                feed_dict[m.inputs_mt_noise_chars] = inputs_mt_noise_chars

                dis_feed_dict[m.inputs_mt_sources] = inputs_mt_sources
                dis_feed_dict[m.inputs_mt_noise_chars] = inputs_mt_noise_chars

                feed_dict[m.inputs_seq_length] = inputs_seq_length
                feed_dict[m.inputs_mt_seq_length] = inputs_mt_seq_length
                dis_feed_dict[m.inputs_mt_seq_length] = inputs_mt_seq_length
                feed_dict[m.keep_rate] = cfg.keep_rate
                dis_feed_dict[m.keep_rate] = cfg.keep_rate
                feed_dict[m.mode] = cfg.mode

                cfg.mode = 'train'

                # update discriminator parameters
                (dis_lr, dis_loss, dis_acc, _) = sess.run((m.dis_lr, m.dis_loss, m.dis_acc, m.dis_train_op), feed_dict=dis_feed_dict)

                (lr, loss, predictions, _, summ) = sess.run((m.lr, m.total_loss, m.predictions, m.train_op, merged_sum), feed_dict=feed_dict)

                loss_li.append(loss)
                dis_loss_li.append(dis_loss)

                step = i + epoch*n_batches
                # add summary
                m.summary_writer.add_summary(summ, step)

                if step % 10 == 1:
                    print('Epoch: %s step: %s lr: %s loss: %s dis_lr: %s dis_loss: %s dis_acc: %s' % (
                        epoch, step, lr, loss, dis_lr, dis_loss, dis_acc))

                if (step > 0 and step <= n_batches and step % int(n_batches/2) == 0) or (step % int(n_batches/4) == 0 and step > n_batches):
                    eval_loss = evaluation(m, sess, valid_data, best_loss, step)
                    curr_mean_loss = eval_loss
                    # decrease learning rate if no improvement and early stopping
                    if curr_mean_loss >= prev_mean_loss and prev_mean_loss != 0.0 and step >= cfg.start_decay_at:
                        loss_add_counts += 1
                        # 如果连续三次loss都increase，则early stopping
                        if loss_add_counts >= 3:
                            print('Early Stopping!!!')
                            exit()
                        if lr > 0.00005:
                            sess.run(m.lr_decay_op)
                    elif curr_mean_loss < prev_mean_loss and prev_mean_loss != 0.0:
                        loss_add_counts = 0

                    mean_dis_loss = np.mean(dis_loss_li)
                    if mean_dis_loss < best_dis_loss:
                        best_dis_loss = mean_dis_loss
                    else:
                        if dis_lr > 0.00005:
                            sess.run(m.dis_lr_decay_op)

                    del dis_loss_li[:]

                    prev_mean_loss = curr_mean_loss

                    if curr_mean_loss < best_loss:
                        best_loss = curr_mean_loss
                        saved_model_file = '{}_{}_{:.3f}.ckpt'.format(epoch, step, best_loss)
                        saved_model_path = os.path.join(model_path, saved_model_file)
                        print('model saved in file : %s' % saved_model_path)
                        saver.save(sess, saved_model_path)

                if step > n_batches:
                    cfg.keep_rate = np.minimum(cfg.keep_rate * 1.01, 1.0)


if __name__ == "__main__":
    train_mt()   # pre-train MT model
    # train()

