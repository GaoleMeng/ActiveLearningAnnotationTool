import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn;
from tensorflow.contrib import legacy_seq2seq
from constants import *


class args():
	def __init__(self):
		self.seq_len = 50;
		self.batch_size = 32;
		self.rnn_size = 100;
		self.input_keep_prob = 0.5;
		self.output_keep_prob = 0.5;
		self.bi_rnn = False;
		self.whetherattention = False;


class Model():
	def __init__(self, args, num_layers=1, haha = True):
		# if not training:
		# 	args.batch_size = 1;
		# 	args.seq_length = MAX_TOKEN_SEQUENCE_LENGTH;
		if haha==True:
			with tf.variable_scope("rnnlm"):
				cell_fn = rnn.BasicLSTMCell;
				self.cell = cell_fn(args.rnn_size);
				self.bi_cell = cell_fn(args.rnn_size)
				print args.seq_len, args.bi_rnn;
				self.cell = rnn.DropoutWrapper(self.cell,input_keep_prob=args.input_keep_prob)
				self.bi_cell = rnn.DropoutWrapper(self.bi_cell,input_keep_prob=args.input_keep_prob)

				if (args.whetherattention == True):
					self.cell = tf.contrib.rnn.AttentionCellWrapper(self.cell, 50, state_is_tuple=True)
					self.bi_cell = tf.contrib.rnn.AttentionCellWrapper(self.bi_cell, 50, state_is_tuple=True)
		else:
			with tf.variable_scope("rnnlm",reuse=True):
				cell_fn = rnn.BasicLSTMCell;
				self.cell = cell_fn(args.rnn_size);
				self.bi_cell = cell_fn(args.rnn_size)
				print args.seq_len, args.bi_rnn;
				# self.cell = rnn.DropoutWrapper(self.cell,input_keep_prob=args.input_keep_prob,output_keep_prob=args.output_keep_prob)
				# self.bi_cell = rnn.DropoutWrapper(self.bi_cell,input_keep_prob=args.input_keep_prob,output_keep_prob=args.output_keep_prob)

				if (args.whetherattention == True):
					self.cell = tf.contrib.rnn.AttentionCellWrapper(self.cell, 50, state_is_tuple=True)
					self.bi_cell = tf.contrib.rnn.AttentionCellWrapper(self.bi_cell, 50, state_is_tuple=True)


class LSTMlogits():
	count = 0;
	def __init__(self, training_LSTM=True):
		self.args = args();
		
		if (training_LSTM == False):
			self.model = Model(self.args, haha=False)
			self.count = 1;
		else:
			self.model = Model(self.args)

		
	def get_embedding(self, doc, W_emb):
		args = self.args;
		truncted_doc = tf.slice(doc, [0, 0], [-1, args.seq_len]);

		self.input_data = truncted_doc;

		#seq_length = tf.placeholders(tf.int32, [tf.shape(doc)[0]])
		#seq_length = tf.random_uniform([tf.shape(doc)[0]], minval = 10, maxval = 100, dtype = tf.int32)

		inputs = tf.nn.embedding_lookup(W_emb, self.input_data)
		print "LSTM model: calling get_embedding building graph: ", self.count

		# inputs = tf.unstack(inputs, axis = 1);
		#outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, None, scope = "rnnlm");

		self.initial_state = self.model.cell.zero_state(tf.shape(doc)[0], tf.float32)
		if (LSTMlogits.count==0):
			if args.bi_rnn == False:
				with tf.variable_scope("rnnlm"):
					outputs, last_state = tf.nn.dynamic_rnn(self.model.cell, inputs, initial_state=self.initial_state);
			else:
				with tf.variable_scope("rnnlm"):
					outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
						self.model.cell, 
						self.model.bi_cell,
						inputs,
						initial_state_fw = self.initial_state,
						initial_state_bw = self.initial_state,
						);
					#outputs = outputs[1];
					outputs = tf.reduce_sum(outputs, 0);
		else:
			if args.bi_rnn == False:
				with tf.variable_scope("rnnlm",reuse=True):
					outputs, last_state = tf.nn.dynamic_rnn(self.model.cell, inputs, initial_state=self.initial_state);
			else:
				with tf.variable_scope("rnnlm",reuse=True):
					outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
						self.model.cell, 
						self.model.bi_cell,
						inputs, 
						initial_state_fw = self.initial_state,
						initial_state_bw = self.initial_state,
						);
					#outputs = outputs[1];
					outputs = tf.reduce_sum(outputs, 0);

		# output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size]);
		self.text_embedding = tf.reduce_sum(outputs, 1);
		self.last_state = last_state;
		
		LSTMlogits.count += 1;
		return self.text_embedding;




