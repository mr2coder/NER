# -*- coding:utf-8 -*-
# @author	: zjun
# @time		: 2017/07/26

import tensorflow as tf

class config(object):
	"""
	配置类
	attrib.:
		hidden_size => 隐藏层数量
		embedding_size => 字向量尺寸
		batch_size => 训练batch_size
		dropout_rate => dropout rate，在lstm中会用到
		layes => 双向lstm层数
		epochs => 迭代次数
	"""
	def __init__(self, kwarg):
		self.hidden_size = kwarg.get('hidden_size')
		self.embedding_size = kwarg.get('embedding_size')
		self.batch_size = kwarg.get('batch_size')
		self.dropout_rate = kwarg.get('dropout_rate')
		self.layes = kwarg.get('layes')
		self.num_epochs = kwarg.get('num_epochs')
		self.vocab_size = kwarg.get('vocab_size')
		self.number_of_classes = kwarg.get('number_of_classes')
		self.sentence_max_len = kwarg.get('sentence_max_len')
		self.optimizer = kwarg.get('optimizer')
		self.learning_rate = kwarg.get('learning_rate')
		self.gradient_clipping_value = kwarg.get('gradient_clipping_value')
		self.UNK = kwarg.get('UNK')
		

class BiLSTM_CRF(object):
	"""
	双向LSTM加CRF模型类
	"""
	def __init__(self, config):
		self.input_words = tf.placeholder(tf.int32,[None,config.sentence_max_len],name='input_words')
		self.input_labels = tf.placeholder(tf.int32,[None,config.sentence_max_len,config.number_of_classes],name='input_labels')
		self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
		self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
		self.config = config

		# 参数初始化
		self.initializer = tf.contrib.layers.xavier_initializer()

		# 字嵌入
		with tf.name_scope("embedding"):
			self.embedding_weights = tf.get_variable(
				name='embedding_weights',
				shape=[config.vocab_size,config.embedding_size],
				initializer=self.initializer)
			self.embedding_words = tf.nn.embedding_lookup(self.embedding_weights,self.input_words)
			lstm_input_drop = self.embedding_words#tf.nn.dropout(self.embedding_words, config.dropout_rate, name='lstm_input_drop')
			# lstm_input_drop_expanded = tf.expand_dims(lstm_input_drop, axis=0, name='lstm_input_drop_expanded')



		with tf.variable_scope("network") as vs:
			outputs = self.BiLSTM(lstm_input_drop)
			lstm_outputs = tf.reshape(outputs,shape=[-1,2 * config.hidden_size])
			# lstm_output_squeezed = tf.squeeze(outputs, axis=0)
			self.lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

		with tf.variable_scope("feedforward_after_lstm") as vs:
			W = tf.get_variable(
				"W1",
				shape=[2 * config.hidden_size, config.hidden_size],
				initializer=self.initializer)
			b = tf.Variable(tf.constant(0.0, shape=[config.hidden_size]), name="bias1")
			outputs = tf.nn.xw_plus_b(lstm_outputs, W, b, name="output_before_tanh")
			outputs = tf.nn.tanh(outputs, name="output_after_tanh")
			# utils_tf.variable_summaries(W)
			# utils_tf.variable_summaries(b)
			# self.token_lstm_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

		with tf.variable_scope("feedforward") as vs:
			W = tf.get_variable(
				"W2",
				shape=[config.hidden_size, config.number_of_classes],
				initializer=self.initializer)
			b = tf.Variable(tf.constant(0.0, shape=[config.number_of_classes]), name="bias2")
			scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
			self.unary_scores = scores
			self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
			# utils_tf.variable_summaries(W)
			# utils_tf.variable_summaries(b)
			# self.feedforward_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

		with tf.variable_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_labels, name='softmax')
			self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss')
		with tf.variable_scope("accuracy"):
			input_labels = tf.reshape(self.input_labels,shape=[-1,config.number_of_classes])
			#测试输入正确性
			# self.test_label = tf.argmax(input_labels, 1)
			#
			correct_predictions = tf.equal(self.predictions, tf.argmax(input_labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		if config.optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(config.learning_rate/(1.0+self.global_step/1000))
		elif config.optimizer == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(config.learning_rate/(1.0+self.global_step/1000))
		elif config.optimizer == 'adadelta':
			self.optimizer = tf.train.AdadeltaOptimizer(config.learning_rate/(1.0+self.global_step/1000))
		else:
			raise ValueError('The lr_method parameter must be either adadelta, adam or sgd.')

		grads_and_vars = self.optimizer.compute_gradients(self.loss)
		if config.gradient_clipping_value:
			grads_and_vars = [(tf.clip_by_value(grad, -config.gradient_clipping_value, config.gradient_clipping_value), var) 
							  for grad, var in grads_and_vars]
		# By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
		# The global step will be automatically incremented by one every time you execute train_op.
		self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

		for g, v in grads_and_vars:
			if g is not None:
				tf.summary.histogram("{}/grad/hist".format(v.name), g)
				tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

		tf.summary.scalar('loss',self.loss)
		tf.summary.scalar('accuracy',self.accuracy)

	def BiLSTM(self,inputs):
		'''
		lstm 
		'''

		def lstm_cell():
		  return tf.contrib.rnn.BasicLSTMCell(
		  self.config.hidden_size, forget_bias=0.5, state_is_tuple=True)
		
		def attn_cell():
			return tf.contrib.rnn.DropoutWrapper(
			lstm_cell(), output_keep_prob=self.config.dropout_rate)

		lstm_fw_cell_m = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in  range(self.config.layes)], 
			state_is_tuple=True)
		lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in  range(self.config.layes)], 
			state_is_tuple=True)

		'''
		with tf.name_scope("BiLSTM"):
			lstm_cell = {}
			initial_state = {}
			for direction in ["forward", "backward"]:
				with tf.variable_scope(direction):
					# LSTM cell
					lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
							self.config.hidden_size, forget_bias=1.0, initializer=self.initializer, state_is_tuple=True)
					# initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
					initial_cell_state = tf.get_variable("initial_cell_state", 
							shape=[1, self.config.hidden_size], dtype=tf.float32, initializer=self.initializer)
					initial_output_state = tf.get_variable("initial_output_state", 
							shape=[1, self.config.hidden_size], dtype=tf.float32, initializer=self.initializer)
					c_states = tf.tile(initial_cell_state, tf.stack([self.config.batch_size, 1]))
					h_states = tf.tile(initial_output_state, tf.stack([self.config.batch_size, 1]))
					initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

			outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
																	lstm_cell["backward"],
																	inputs,
																	dtype=tf.float32,
																	sequence_length=self.sequence_length,
																	initial_state_fw=initial_state["forward"],
																	initial_state_bw=initial_state["backward"])
			'''
		outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, 
																lstm_bw_cell_m, 
																inputs, 
																sequence_length=self.sequence_length,
																dtype=tf.float32)

		outputs_forward, outputs_backward = outputs
		output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')

		return output

	def CRF(self):
		'''
		'''
		small_score = -1000.0
		large_score = 0.0
		sequence_length = tf.shape(self.unary_scores)[0]
		unary_scores_with_start_and_end = tf.concat([self.unary_scores, tf.tile( tf.constant(small_score, shape=[1, 2]) , [sequence_length, 1])], 1)
		start_unary_scores = [[small_score] * dataset.number_of_classes + [large_score, small_score]]
		end_unary_scores = [[small_score] * dataset.number_of_classes + [small_score, large_score]]
		self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
		start_index = dataset.number_of_classes
		end_index = dataset.number_of_classes + 1
		input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.input_label_indices_flat, tf.constant(end_index, shape=[1]) ], 0)

		# Apply CRF layer
		sequence_length = tf.shape(self.unary_scores)[0]
		sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
		unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
		input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')
		if self.verbose: print('unary_scores_expanded: {0}'.format(unary_scores_expanded))
		if self.verbose: print('input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_batch))
		if self.verbose: print("sequence_lengths: {0}".format(sequence_lengths))
		# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
		# Compute the log-likelihood of the gold sequences and keep the transition params for inference at test time.
		self.transition_parameters=tf.get_variable(
			"transitions",
			shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
			initializer=initializer)
		utils_tf.variable_summaries(self.transition_parameters)
		log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
			unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths, transition_params=self.transition_parameters)
		self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
		self.accuracy = tf.constant(1)

	

