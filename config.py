#!/usr/bin/python
# -*- coding:utf8 -*-

import tensorflow as tf

tf.set_random_seed(10)

flags = tf.app.flags

flags.DEFINE_string('model_dir', './models/', 'choose pre-trained model to init the training model')
flags.DEFINE_string('summary_dir', './summary/', 'summary dir')
flags.DEFINE_integer('eval_every', 500, 'evaluate the model every 500 epochs')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('decay_steps', 50, 'decay every 50 steps with decay rate:lr_decay_rate')
flags.DEFINE_float('keep_rate', 0.6, '')
flags.DEFINE_float('lr', 0.0003, '')
flags.DEFINE_float('dis_lr', 0.0005, 'Discriminator learning rate')
flags.DEFINE_float('mt_lr', 0.001, 'pre-trained MT model\'s learning rate')
flags.DEFINE_float('lr_decay_rate', 0.99, '')
flags.DEFINE_integer('start_decay_at', 2000, '')
flags.DEFINE_float('grad_clip', 5.0, '')
flags.DEFINE_string('optimizer', 'adam', '')
flags.DEFINE_string('device', '/gpu:2', '')
flags.DEFINE_float('stddev', 0.01, '')
flags.DEFINE_integer('enc_hidden_size', 256, '')
flags.DEFINE_integer('dec_hidden_size', 512, 'dec_hidden_size*3==enc_hidden_size*6')

flags.DEFINE_integer('char_size', 1465, '')
flags.DEFINE_integer('sample_size', 1465, '')
flags.DEFINE_integer('char_vec_dim', 100, '')
flags.DEFINE_float('inputs_chars_length', 30, '')
flags.DEFINE_float('dec_chars_length', 30, '')
flags.DEFINE_integer('num_heads', 1, '')
flags.DEFINE_integer('num_epochs', 100, '')

flags.DEFINE_float('reg_ae', 1.0, '')
flags.DEFINE_float('reg_mt', 1.0, '')
flags.DEFINE_float('reg_adv', 1.0, '')
flags.DEFINE_float('alpha', 0.2, 'leaky_relu hyper-param')
flags.DEFINE_string('mode', 'train', '')

flags.DEFINE_string('data_path', './data/', 'data path')

cfg = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
