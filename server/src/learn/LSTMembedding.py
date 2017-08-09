# this model calculate the embedding of the doc by using lstm model.


import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn;
from tensorflow.contrib import legacy_seq2seq
from constants import *


class args():
	def __init__(self):
		self.seq_len = 40;
		self.batch_size = 32;
		self.rnn_size = 100;
		self.input_keep_prob = 0.5;
		self.output_keep_prob = 0.5;

class Model():
	def __init__(self, args, num_layers = 1):

		cells = []
		cell_fn = rnn.BasicLSTMCell;
		for _ in range(num_layers):
			cell = cell_fn(args.rnn_size)
			cell = rnn.DropoutWrapper(cell,
				input_keep_prob=args.input_keep_prob,
				output_keep_prob=args.output_keep_prob)

			cells.append(cell)
		self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)



class LSTMlogits():
	def __init__(self, training_LSTM=True):
		self.args = args();
		self.model = Model(self.args)

	def get_embedding(self, doc, W_emb):
		args = self.args;
		truncted_doc = tf.slice(doc, [0, 0], [-1, args.seq_len]);
#		truncted_doc = tf.reshape(doc, [-1, args.seq_len])


		self.input_data = truncted_doc;
		self.initial_state = self.model.cell.zero_state(tf.shape(truncted_doc)[0], tf.float32)

		inputs = tf.nn.embedding_lookup(W_emb, self.input_data)

		# inputs = tf.unstack(inputs, axis = 1);

		#outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, None, scope = "rnnlm");
		outputs, last_state = tf.nn.dynamic_rnn(self.model.cell, inputs, initial_state=self.initial_state ,scope = "rnnlm");

		# output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size]);
		self.text_embedding = tf.reduce_sum(outputs, 1);
		#self.text_embedding = tf.reshape(self.text_embedding, [tf.shape(self.text_embedding)[0]/(MAX_TOKEN_SEQUENCE_LENGTH/args.seq_len), MAX_TOKEN_SEQUENCE_LENGTH/args.seq_len, args.rnn_size])
		# self.text_embedding = tf.slice(self.tmp_text_embedding, [0, 0], [, args.seq_len]);
		#self.text_embedding = tf.reduce_sum(self.text_embedding, 1);
		# self.reshaped = tf.reshape(self.text_embedding, [tf.shape(doc)[0], MAX_TOKEN_SEQUENCE_LENGTH/args.seq_len, args.rnn_size])
		# self.text_embedding = tf.reduce_sum(self.reshaped, 1);
		
		self.last_state = last_state;
		return self.text_embedding;








	