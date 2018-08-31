#!/usr/bin/python
# -*- coding:utf8 -*-
"""
用机器翻译模型来完成数据增强
论文：《Unsupervised Machine Translation Using Monolingual Corpora Only》
link: https://arxiv.org/pdf/1711.00043.pdf
"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(10)
np.random.seed(10)

from my_seq2seq import my_stack_embedding_attention_seq2seq
from config import cfg


class Model(object):
	def __init__(self, char_size, char_embedding_size):
		self.lr = tf.Variable(float(cfg.lr), trainable=False, dtype=tf.float32)
		self.lr_decay_op = self.lr.assign(self.lr * cfg.lr_decay_rate)
		self.dis_lr = tf.Variable(float(cfg.dis_lr), trainable=False, dtype=tf.float32)
		self.dis_lr_decay_op = self.dis_lr.assign(self.dis_lr * cfg.lr_decay_rate)
		self.mt_lr = tf.Variable(float(cfg.mt_lr), trainable=False, dtype=tf.float32)
		self.mt_lr_decay_op = self.mt_lr.assign(self.mt_lr * cfg.lr_decay_rate)
		self.keep_rate = tf.placeholder(tf.float32, (), name='keep_rate')
		self.global_step = tf.Variable(0, trainable=False)
		self.dis_global_step = tf.Variable(0, trainable=False)
		self.mt_global_step = tf.Variable(0, trainable=False)
		self.reg_ae = tf.Variable(float(cfg.reg_ae), trainable=False)
		self.reg_mt = tf.Variable(float(cfg.reg_mt), trainable=False)
		self.reg_adv = tf.Variable(float(cfg.reg_adv), trainable=False)
		self.mode = tf.placeholder(tf.string, (), name='mode')

		self.inputs_noise_chars = tf.placeholder(tf.int32, [None, cfg.inputs_chars_length]) # 输入加noise文本
		self.inputs_mt_noise_chars = tf.placeholder(tf.int32, [None, cfg.inputs_chars_length]) # 输入加noise的翻译文本
		self.inputs_sources = tf.placeholder(tf.int32, [None]) # 输入源,person1 or person2
		self.inputs_mt_sources = tf.placeholder(tf.int32, [None])  # mt输入源, 与inputs_source互斥
		self.inputs_mask = tf.placeholder(tf.float32, [None, cfg.inputs_chars_length])
		self.inputs_seq_length = tf.placeholder(tf.int32, [None])
		self.inputs_mt_mask = tf.placeholder(tf.float32, [None, cfg.inputs_chars_length])
		self.inputs_mt_seq_length = tf.placeholder(tf.int32, [None])

		self.dec_inputs = [tf.placeholder(tf.int32, [None], name='dec_inputs_{}'.format(t)) for t in range(cfg.dec_chars_length)]
		self.dec_targets = [tf.placeholder(tf.int32, [None], name='dec_targets_{}'.format(t)) for t in range(cfg.dec_chars_length)]
		self.dec_mask = [tf.placeholder(tf.float32, [None], name='dec_masks_{}'.format(t)) for t in range(cfg.dec_chars_length)]

		with tf.variable_scope("char_embedding_layer"):
			char_embedding = tf.get_variable('char_embedding', [char_size, char_embedding_size], tf.float32)

		zero_mask = tf.constant([0 if i == 0 else 1 for i in range(char_size)], dtype=tf.float32, shape=[char_size, 1])
		self.char_embedding = char_embedding * zero_mask

		inputs_noise_chars_embedding = tf.nn.embedding_lookup(self.char_embedding, self.inputs_noise_chars)
		inputs_mt_noise_chars_embedding = tf.nn.embedding_lookup(self.char_embedding, self.inputs_mt_noise_chars)

		output_projection = None
		softmax_loss_function = None

		proj_w_t = tf.get_variable("proj_w", [cfg.char_size, cfg.dec_hidden_size], dtype=tf.float32)
		proj_w = tf.transpose(proj_w_t)
		proj_b = tf.get_variable("proj_b", [cfg.char_size], dtype=tf.float32)
		output_projection = (proj_w, proj_b)

		# Sampled softmax only makes sense if sample size less than vocabulary size.
		if 0 < cfg.sample_size and cfg.sample_size < cfg.char_size:
			def sampled_loss(labels, logits):
				labels = tf.reshape(labels, [-1, 1])
				local_w_t = tf.cast(proj_w_t, tf.float32)
				local_b = tf.cast(proj_b, tf.float32)
				local_logits = tf.cast(logits, tf.float32)

				return tf.cast(tf.nn.sampled_softmax_loss(weights=local_w_t, biases=local_b, labels=labels, inputs=local_logits, num_sampled=cfg.sample_size, num_classes=cfg.char_size), tf.float32)

			softmax_loss_function = sampled_loss

		stacked_fw_rnn = []
		stacked_bw_rnn = []
		for i in range(3):
			stacked_fw_rnn.append(
				tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(cfg.enc_hidden_size), output_keep_prob=self.keep_rate))
			stacked_bw_rnn.append(
				tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(cfg.enc_hidden_size), output_keep_prob=self.keep_rate))

		stacked_dec_rnn_li = []
		for i in range(3):
			stacked_dec_rnn_li.append(
				tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(cfg.dec_hidden_size), output_keep_prob=self.keep_rate))

		stacked_dec_rnn = tf.contrib.rnn.MultiRNNCell(stacked_dec_rnn_li)

		with tf.variable_scope('Encoder', dtype=tf.float32, reuse=None):
			enc_outputs, fw_hidden_state_li, bw_hidden_state_li = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
				cells_fw=stacked_fw_rnn, cells_bw=stacked_bw_rnn, inputs=inputs_noise_chars_embedding,
				sequence_length=self.inputs_seq_length, dtype=tf.float32
			)
			# [batch_size, 3*enc_hidden_size]
			fw_hidden_state = tf.concat(fw_hidden_state_li, 1)
			bw_hidden_state = tf.concat(bw_hidden_state_li, 1)
			assert(fw_hidden_state.get_shape()[1] == 3 * cfg.enc_hidden_size)
			assert(bw_hidden_state.get_shape()[1] == 3 * cfg.enc_hidden_size)

		with tf.variable_scope('Encoder', dtype=tf.float32, reuse=True):
			mt_enc_outputs, mt_fw_hidden_state_li, mt_bw_hidden_state_li = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
				cells_fw=stacked_fw_rnn, cells_bw=stacked_bw_rnn, inputs=inputs_mt_noise_chars_embedding,
				sequence_length=self.inputs_mt_seq_length, dtype=tf.float32
			)
			# [batch_size, 3*enc_hidden_size]
			mt_fw_hidden_state = tf.concat(mt_fw_hidden_state_li, 1)
			mt_bw_hidden_state = tf.concat(mt_bw_hidden_state_li, 1)
			assert(mt_fw_hidden_state.get_shape()[1] == 3 * cfg.enc_hidden_size)
			assert(mt_bw_hidden_state.get_shape()[1] == 3 * cfg.enc_hidden_size)

		with tf.variable_scope('Discriminator'):
			# [batch_size, 6*enc_hidden_size]
			concat_hidden_state = tf.concat([mt_fw_hidden_state, mt_bw_hidden_state], 1)
			dis_w_1 = tf.get_variable("dis_w_1", [cfg.enc_hidden_size*6, cfg.enc_hidden_size], dtype=tf.float32)
			dis_b_1 = tf.get_variable("dis_b_1", [cfg.enc_hidden_size], dtype=tf.float32)
			dis_w_2 = tf.get_variable("dis_w_2", [cfg.enc_hidden_size, int(cfg.enc_hidden_size/2)], dtype=tf.float32)
			dis_b_2 = tf.get_variable("dis_b_2", [int(cfg.enc_hidden_size/2)], dtype=tf.float32)
			dis_w_3 = tf.get_variable("dis_w_3", [int(cfg.enc_hidden_size/2), 2], dtype=tf.float32)
			dis_b_3 = tf.get_variable("dis_b_3", [2], dtype=tf.float32)

			tmp_1 = tf.nn.xw_plus_b(tf.stop_gradient(concat_hidden_state), dis_w_1, dis_b_1)
			tmp_1 = tf.maximum(cfg.alpha*tmp_1, tmp_1) # leaky_relu activations
			tmp_2 = tf.nn.xw_plus_b(tmp_1, dis_w_2, dis_b_2)
			tmp_2 = tf.maximum(cfg.alpha*tmp_2, tmp_2)
			dis_logits = tf.nn.xw_plus_b(tmp_2, dis_w_3, dis_b_3)

			adv_tmp_1 = tf.nn.xw_plus_b(concat_hidden_state, tf.stop_gradient(dis_w_1), tf.stop_gradient(dis_b_1))
			adv_tmp_1 = tf.maximum(cfg.alpha * adv_tmp_1, adv_tmp_1)  # leaky_relu activations
			adv_tmp_2 = tf.nn.xw_plus_b(adv_tmp_1, tf.stop_gradient(dis_w_2), tf.stop_gradient(dis_b_2))
			adv_tmp_2 = tf.maximum(cfg.alpha * adv_tmp_2, adv_tmp_2)
			adv_logits = tf.nn.xw_plus_b(adv_tmp_2, tf.stop_gradient(dis_w_3), tf.stop_gradient(dis_b_3))

			# [batch_size, 2]
			self.dis_probs = tf.nn.softmax(dis_logits)
			self.adv_probs = tf.nn.softmax(adv_logits)

			dis_preds = tf.cast(tf.reshape(tf.argmax(self.dis_probs, axis=1), [cfg.batch_size]), tf.int32)

			self.dis_acc = tf.reduce_mean(tf.cast(tf.equal(dis_preds, self.inputs_mt_sources), tf.float32))

			self.dis_loss = -tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.dis_probs, [-1]), [2*b+self.inputs_mt_sources[b] for b in range(cfg.batch_size)])))
			self.adv_loss = -tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.adv_probs, [-1]), [2*b+self.inputs_sources[b] for b in range(cfg.batch_size)])))

		with tf.variable_scope('Decoder', reuse=None):
			# [seq_len, batch_size, 2*enc_hidden_size]
			trans_enc_outputs = tf.transpose(enc_outputs, perm=[1, 0, 2])

			# ([batch_size, 2*enc_hidden_size])
			tuple_state = tuple([tf.concat([fw_hidden_state_li[t], bw_hidden_state_li[t]], 1) for t in range(len(fw_hidden_state_li))])

			# [seq_len, [batch_size, enc_hidden_size*2]]
			concat_enc_outputs = [trans_enc_outputs[t] for t in range(cfg.inputs_chars_length)]

			# stacked RNN Decoder with attention
			dec_outputs, dec_states = my_stack_embedding_attention_seq2seq(concat_enc_outputs,
										 tuple_state, self.dec_inputs,
										 stacked_dec_rnn, cfg.char_size,
										 cfg.char_vec_dim,
										 output_projection=output_projection if output_projection else None,
										 feed_previous=bool(cfg.mode == 'test'),
										 dtype=tf.float32)

			dec_logits = [tf.nn.xw_plus_b(dec_outputs[t], proj_w, proj_b) for t in range(cfg.dec_chars_length)]

			# when not using sampled softmax
			if cfg.sample_size == cfg.char_size:
				# [seq_len, [batch_size, char_size]]
				self.ae_dec_logits = dec_logits
			else:
				# [seq_len, [batch_size, char_size]]
				self.ae_dec_logits = dec_outputs

		with tf.variable_scope('Decoder', reuse=True):
			# [seq_len, batch_size, 2*enc_hidden_size]
			trans_mt_enc_outputs = tf.transpose(mt_enc_outputs, perm=[1, 0, 2])

			# ([batch_size, 2*enc_hidden_size])
			mt_tuple_state = tuple([tf.concat([mt_fw_hidden_state_li[t], mt_bw_hidden_state_li[t]], 1) for t in range(len(mt_fw_hidden_state_li))])

			# [seq_len, [batch_size, enc_hidden_size*2]]
			concat_mt_enc_outputs = [trans_mt_enc_outputs[t] for t in range(cfg.inputs_chars_length)]

			# stacked RNN Decoder with attention
			dec_mt_outputs, dec_mt_states = my_stack_embedding_attention_seq2seq(concat_mt_enc_outputs,
										       mt_tuple_state,
										       self.dec_inputs, stacked_dec_rnn,
										       cfg.char_size, cfg.char_vec_dim,
										       output_projection=output_projection if output_projection else None,
										       feed_previous=bool(cfg.mode == 'test'),
										       dtype=tf.float32)

			valid_dec_mt_outputs, valid_dec_mt_states = my_stack_embedding_attention_seq2seq(concat_mt_enc_outputs,
										       mt_tuple_state,
										       self.dec_inputs, stacked_dec_rnn,
										       cfg.char_size, cfg.char_vec_dim,
										       output_projection=output_projection if output_projection else None,
										       feed_previous=True,
										       dtype=tf.float32)

			dec_mt_logits = [tf.nn.xw_plus_b(dec_mt_outputs[t], proj_w, proj_b) for t in range(cfg.dec_chars_length)]
			valid_dec_mt_logits = [tf.nn.xw_plus_b(valid_dec_mt_outputs[t], proj_w, proj_b) for t in range(cfg.dec_chars_length)]

			# when not using sampled softmax
			if cfg.sample_size == cfg.char_size:
				# [seq_len, [batch_size, char_size]]
				self.mt_dec_logits = dec_mt_logits
			else:
				# [seq_len, [batch_size, char_size]]
				self.mt_dec_logits = dec_mt_outputs

		self.ae_loss = tf.contrib.legacy_seq2seq.sequence_loss(self.ae_dec_logits, self.dec_targets, self.dec_mask, softmax_loss_function=softmax_loss_function)
		self.mt_loss = tf.contrib.legacy_seq2seq.sequence_loss(self.mt_dec_logits, self.dec_targets, self.dec_mask, softmax_loss_function=softmax_loss_function)

		self.total_loss = self.reg_ae * self.ae_loss + self.reg_mt * self.mt_loss + self.reg_adv * self.adv_loss

		# [seq_len, [batch_size, char_size]]
		p_mt_over_vocab = [tf.nn.softmax(dec_mt_logits[t]) for t in range(cfg.dec_chars_length)]
		valid_p_mt_over_vocab = [tf.nn.softmax(valid_dec_mt_logits[t]) for t in range(cfg.dec_chars_length)]

		# [seq_len, [batch_size,]]
		predictions = [tf.argmax(p_mt_over_vocab[t], axis=1) for t in range(cfg.dec_chars_length)]
		valid_predictions = [tf.argmax(valid_p_mt_over_vocab[t], axis=1) for t in range(cfg.dec_chars_length)]
		# [seq_len, batch_size]
		self.predictions = tf.reshape(predictions, [cfg.dec_chars_length, cfg.batch_size])
		self.valid_predictions = tf.reshape(valid_predictions, [cfg.dec_chars_length, cfg.batch_size])

		with tf.name_scope('backpropagation'):
			if cfg.optimizer == 'adam':
				optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
			elif cfg.optimizer == 'rmsprop':
				optim = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.0, epsilon=1e-08)
			else:
				print('Please check the hyparameter optimizer!!!')
				exit()

			dis_optim = tf.train.RMSPropOptimizer(learning_rate=self.dis_lr, decay=0.9, momentum=0.0, epsilon=1e-08)
			mt_optim = tf.train.AdamOptimizer(learning_rate=self.mt_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)

			# clip gradients
			params_to_train = tf.trainable_variables()

			mt_gradients = tf.gradients(self.mt_loss, params_to_train)
			dis_gradients = tf.gradients(self.dis_loss, params_to_train)
			gradients = tf.gradients(self.total_loss, params_to_train)
			mt_clipped_gradients, mt_norm = tf.clip_by_global_norm(mt_gradients, cfg.grad_clip)
			dis_clipped_gradients, dis_norm = tf.clip_by_global_norm(dis_gradients, cfg.grad_clip)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, cfg.grad_clip)

			self.mt_train_op = mt_optim.apply_gradients(zip(mt_clipped_gradients, params_to_train), global_step=self.mt_global_step)
			self.dis_train_op = dis_optim.apply_gradients(zip(dis_clipped_gradients, params_to_train), global_step=self.dis_global_step)
			self.train_op = optim.apply_gradients(zip(clipped_gradients, params_to_train), global_step=self.global_step)

		# summary
		tf.summary.scalar('total_loss', self.total_loss)
		tf.summary.scalar('dis_loss', self.dis_loss)
		tf.summary.scalar('dis_acc', self.dis_acc)
		tf.summary.scalar('mt_loss', self.mt_loss)
		self.summary_writer = tf.summary.FileWriter(cfg.summary_dir)

