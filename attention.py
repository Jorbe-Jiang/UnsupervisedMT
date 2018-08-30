#!/usr/bin/python
# -*- coding:utf8 -*-
"""
implementation of directional self attention, self attention, windows soft attention and standard soft attention
Reference paper & code: https://arxiv.org/pdf/1709.04696.pdf [directional self attention]
			https://openreview.net/pdf?id=H1cWzoxA- [a simple introduction of self attention]
			https://github.com/taoshen58/DiSAN/blob/master/SST_disan/src/nn_utils/disan.py

"""
import tensorflow as tf

tf.set_random_seed(10)


def mask_for_high_rank(val, val_mask, name=None):
	val_mask = tf.expand_dims(val_mask, -1)
	return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def exp_mask_for_high_rank(val, val_mask, name=None):
	val_mask = tf.expand_dims(val_mask, -1)
	return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * -1e30, name=name or 'exp_mask_for_high_rank')


def self_attention(inputs, inputs_mask, scope=None, time_major=False, return_alphas=False):
	"""
	inputs: [batch_size, time_steps, hidden_size] or ([batch_size, time_steps, hidden_size], [batch_size, time_steps, hidden_size])
	inputs_mask: [batch_size, time_steps]
	scope: variable scope
	time_major: [T, B, D] or [B, T, D]
	return_alphas: return attention score for visualization

	Returns:
		[batch_size, time_steps, hidden_size]
	"""
	if isinstance(inputs, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs
		inputs = tf.concat(inputs, 2)

	if time_major:
		# [T, B, D] => [B, T, D]
		inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

	def scaled_tanh(x, scale=5.):
		return scale * tf.nn.tanh(1./scale*x)

	batch_size, time_steps, _hidden_size = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
	hidden_size = inputs.get_shape()[2]

	with tf.variable_scope(scope or 'self_attention'):
		# mask generation
		time_steps_indices = tf.range(time_steps, dtype=tf.int32)
		ts_col, ts_row = tf.meshgrid(time_steps_indices, time_steps_indices)

		mask = tf.cast(tf.diag(-tf.ones([time_steps], tf.int32)) + 1, dtype=tf.bool)

		# [batch_size, time_steps, time_steps]
		mask_tile = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1])
		# [batch_size, time_steps, time_steps]
		inputs_mask_tile = tf.tile(tf.expand_dims(inputs_mask, 1), [1, time_steps, 1])
		# [batch_size, time_steps, time_steps]
		attn_mask = tf.logical_and(mask_tile, inputs_mask_tile)

		# [batch_size, time_steps, hidden_size]
		inputs_map = tf.layers.dense(inputs, hidden_size, activation=tf.nn.relu, name='inputs_map')

		# [batch_size, time_steps, time_steps, hidden_size]
		inputs_map_tile = tf.tile(tf.expand_dims(inputs_map, 1), [1, time_steps, 1, 1])

		# attention
		with tf.variable_scope('attention'):
			f_bias = tf.get_variable('f_bias', [hidden_size], tf.float32, tf.constant_initializer(0.0))

			# [batch_size, time_steps, hidden_size]
			dependent = tf.layers.dense(inputs_map, hidden_size, name='dependent_linear')
			# [batch_size, 1, time_steps, hidden_size]
			dependent_epd = tf.expand_dims(dependent, 1)
			# [batch_size, 1, time_steps, hidden_size]
			head = tf.layers.dense(inputs_map, hidden_size, name='head_linear')
			# [batch_size, time_steps, 1, hidden_size]
			head_epd = tf.expand_dims(head, 2)

			# [batch_size, time_steps, time_steps, hidden_size]
			logits = scaled_tanh(dependent_epd+head_epd+f_bias, 5.0)

			logits_masked = exp_mask_for_high_rank(logits, attn_mask)
			# [batch_size, time_steps, time_steps, hidden_size]
			attn_score = tf.nn.softmax(logits_masked, 2)
			attn_score = mask_for_high_rank(attn_score, attn_mask)

			# [batch_size, time_steps, hidden_size]
			output = tf.reduce_sum(attn_score*inputs_map_tile, 2)
			output = mask_for_high_rank(output, inputs_mask)

		if not return_alphas:
			return output
		else:
			return output, attn_score


def directional_self_attention(inputs, inputs_mask, direction=None, scope=None, time_major=False, return_alphas=False):
	"""
	inputs: [batch_size, time_steps, hidden_size] or ([batch_size, time_steps, hidden_size], [batch_size, time_steps, hidden_size])
	inputs_mask: [batch_size, time_steps]
	direction: forward or backward or None
	scope: variable scope
	time_major: [T, B, D] or [B, T, D]
	return_alphas: return attention score for visualization

	Returns:
		[batch_size, time_steps, hidden_size]
	"""
	if isinstance(inputs, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs
		inputs = tf.concat(inputs, 2)

	if time_major:
		# [T, B, D] => [B, T, D]
		inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

	def scaled_tanh(x, scale=5.):
		return scale * tf.nn.tanh(1./scale*x)

	batch_size, time_steps, _hidden_size = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
	hidden_size = inputs.get_shape()[2]

	with tf.variable_scope(scope or 'directional_self_attention_%s' %direction or 'dir_self_attn'):
		# mask generation
		time_steps_indices = tf.range(time_steps, dtype=tf.int32)
		ts_col, ts_row = tf.meshgrid(time_steps_indices, time_steps_indices)

		if direction is None:
			direction_mask = tf.cast(tf.diag(-tf.ones([time_steps], tf.int32))+1, dtype=tf.bool)
		else:
			if direction == 'forward':
				direction_mask = tf.greater(ts_row, ts_col)
			else:
				direction_mask = tf.greater(ts_col, ts_row)
		# [batch_size, time_steps, time_steps]
		direction_mask_tile = tf.tile(tf.expand_dims(direction_mask, 0), [batch_size, 1, 1])
		# [batch_size, time_steps, time_steps]
		inputs_mask_tile = tf.tile(tf.expand_dims(inputs_mask, 1), [1, time_steps, 1])
		# [batch_size, time_steps, time_steps]
		attn_mask = tf.logical_and(direction_mask_tile, inputs_mask_tile)

		# [batch_size, time_steps, hidden_size]
		inputs_map = tf.layers.dense(inputs, hidden_size, activation=tf.nn.relu, name='inputs_map')

		# [batch_size, time_steps, time_steps, hidden_size]
		inputs_map_tile = tf.tile(tf.expand_dims(inputs_map, 1), [1, time_steps, 1, 1])

		# attention
		with tf.variable_scope('attention'):
			f_bias = tf.get_variable('f_bias', [hidden_size], tf.float32, tf.constant_initializer(0.0))

			# [batch_size, time_steps, hidden_size]
			dependent = tf.layers.dense(inputs_map, hidden_size, name='dependent_linear')
			# [batch_size, 1, time_steps, hidden_size]
			dependent_epd = tf.expand_dims(dependent, 1)
			# [batch_size, 1, time_steps, hidden_size]
			head = tf.layers.dense(inputs_map, hidden_size, name='head_linear')
			# [batch_size, time_steps, 1, hidden_size]
			head_epd = tf.expand_dims(head, 2)

			# [batch_size, time_steps, time_steps, hidden_size]
			logits = scaled_tanh(dependent_epd+head_epd+f_bias, 5.0)

			logits_masked = exp_mask_for_high_rank(logits, attn_mask)
			# [batch_size, time_steps, time_steps, hidden_size]
			attn_score = tf.nn.softmax(logits_masked, 2)
			attn_score = mask_for_high_rank(attn_score, attn_mask)

			# [batch_size, time_steps, hidden_size]
			attn_result = tf.reduce_sum(attn_score*inputs_map_tile, 2)

		with tf.variable_scope('output'):
			o_bias = tf.get_variable('o_bias', [hidden_size], tf.float32, tf.constant_initializer(0.0))

			gate_input = tf.layers.dense(inputs_map, hidden_size, name='gate_input_1') + tf.layers.dense(attn_result, hidden_size, name='gate_input_2')

			# input gate
			fusion_gate = tf.nn.sigmoid(gate_input+o_bias)

			output = fusion_gate * inputs_map + (1-fusion_gate)*attn_result
			output = mask_for_high_rank(output, inputs_mask)

		if not return_alphas:
			return output
		else:
			return output, attn_score


def split_wind_inputs(inputs, window_size, stride):
	"""
	inputs: [batch_size, time_steps, hidden_size]
	window_size: window size
	stride: step size

	Returns:
		[batch_size, new_time_steps, new_hidden_size]
		if no concatenate, then new_hidden_size==hidden_size
		else new_hidden_size==window_size*hidden_size
	"""
	inputs = tf.transpose(inputs, [1, 0, 2]) # [time_steps, batch_size, hidden_size]
	time_steps = inputs.shape[0].value
	hidden_size = inputs.shape[2].value
	new_time_steps = int((time_steps - window_size)/stride) + 1

	indices = []
	for i in range(new_time_steps):
		tmp_indices = []
		for j in range(window_size):
			tmp_indices.append([i*stride+j])
		indices.append(tmp_indices)

	# [new_time_steps, window_size, batch_size, hidden_size]
	wind_inputs = tf.gather_nd(inputs, indices)
	# [new_time_steps, batch_size, window_size*hidden_size] concatenate all window size vec
	#wind_inputs = tf.reshape(tf.transpose(wind_inputs, [0, 2, 1, 3]), shape=(new_time_steps, -1, window_size*hidden_size))
	# [new_time_steps, batch_size, hidden_size] element-wise sum all window size vec
	wind_inputs = tf.reduce_sum(wind_inputs, axis=1)

	# [batch_size, new_time_steps, new_hidden_size]
	wind_inputs = tf.transpose(wind_inputs, [1, 0, 2])

	return wind_inputs


def wind_attention(inputs, inputs_mask, inputs_seq_len, attention_size, window_size=2, stride=1, time_major=False, return_alphas=False):
	"""
	    inputs: The Attention inputs.
		Matches outputs of RNN/Bi-RNN layer (not final state):
			In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            inputs_mask: [batch_size, time_steps]
            inputs_seq_len: [batch_size]
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
	    window_size:
        	window size
            stride:
        	step size
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
	"""
	if isinstance(inputs, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs
		inputs = tf.concat(inputs, 2)

	if time_major:
		# [T, B, D] => [B, T, D]
		inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

	# [B, T_new, D_new]
	wind_inputs = split_wind_inputs(inputs, window_size, stride)
	hidden_size = wind_inputs.shape[2].value

	time_steps = wind_inputs.shape[1].value
	# [batch_size]
	new_seq_len = tf.cast((inputs_seq_len-window_size) / stride, tf.int32) + 1
	# [batch_size, T_new]
	new_inputs_mask = tf.sequence_mask(new_seq_len, time_steps)

	# Trainable parameters
	w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
	b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

	with tf.name_scope('v'):
		# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
		# the shape of `v` is (B,T_new,D_new)*(D_new,A)=(B,T_new,A), where A=attention_size
		v = tf.tanh(tf.tensordot(wind_inputs, w_omega, axes=1) + b_omega)

	# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
	vu = tf.tensordot(v, u_omega, axes=1, name='vu') # [B, T_new]
	# [B, T_new]
	masked_vu = tf.add(vu, (1 - tf.cast(new_inputs_mask, tf.float32)) * -1e30, name='mask_vu')
	alphas = tf.nn.softmax(masked_vu, name='alphas') # [B, T_new]

	# Output of (Bi-)RNN is reduced with attention vector; the result has (B, D_new) shape
	output = tf.reduce_sum(wind_inputs * tf.expand_dims(alphas, -1), 1) # [B, D_new]


	if not return_alphas:
		return output
	else:
		return output, alphas


def attention(inputs, inputs_mask, attention_size, time_major=False, return_alphas=False):
	"""
	Args:
		inputs: The Attention inputs.
		Matches outputs of RNN/Bi-RNN layer (not final state):
			In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
                inputs_mask: [batch_size, time_steps]
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
	"""
	if isinstance(inputs, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs
		inputs = tf.concat(inputs, 2)

	if time_major:
		# [T, B, D] => [B, T, D]
		inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

	hidden_size = inputs.shape[2].value # D value: hidden size of the RNN layer

	# Trainable parameters
	w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
	b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

	with tf.name_scope('v'):
		# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
		# the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
		v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

	# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
	vu = tf.tensordot(v, u_omega, axes=1, name='vu') # [B, T]
	# [B, T]
	masked_vu = tf.add(vu, (1-tf.cast(inputs_mask, tf.float32))*-1e30, name='mask_vu')
	alphas = tf.nn.softmax(masked_vu, name='alphas') # [B, T]

	# Output of (Bi-)RNN is reduced with attention vector; the result has (B, D) shape
	output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1) # [B, D]

	if not return_alphas:
		return output
	else:
		return output, alphas
