#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import os, codecs
import numpy as np
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # mute the CPU acceleration warnings
import tensorflow as tf

from collections import Counter
import scipy.sparse as sp

class MySparseData(object):
	def __init__(self):
		self.dict = {}
		self.shape = []

def get_feature_sparse(documents, feat_labels, feat_tasks):
	words = [i for l in documents.values() for i in l]
	word_freq = sorted(Counter(words).items(), key = lambda x: x[1], reverse = True) # we can cut features here
	feat_idx = dict(zip([e[0] for e in word_freq], range(len(word_freq)) ))
	
	# add other features
	feat_i = len(feat_idx)
	for k in feat_labels.keys() + feat_tasks:
		if k not in feat_idx:
			feat_idx[k] = feat_i
			feat_i += 1
	return feat_idx

def featurize_sparse(documents, feat_idx):
	inst_idx = dict(zip(sorted(documents.keys()),  range(len(documents)) ))
	# sparse: bag of words encoding: return sparse data (counts)
	m = MySparseData()
	m.shape = [len(inst_idx), len(feat_idx)]
	for k, v in documents.items():
		for f in v:
			if f in feat_idx: # filter possibly unknown features
				index = (inst_idx[k], feat_idx[f])
				if index not in m.dict:
					m.dict[index] = 1.
				else: # counts
					m.dict[index] += 1.
	return inst_idx, m

def get_label_idx(inst_labels, feat_labels):
	label_idx = {}
	label_i = 0
	for l_dict in inst_labels.values():
		for k, v in l_dict.items():
			if k not in label_idx:
				label_idx[k] = label_i
				label_i += 1
	return label_idx

# L x N sparse matrix
def get_i_mask(inst_idx, inst_labels):
	m = MySparseData()
	m.shape = [len(inst_labels), len(inst_idx)]
	row_idx = 0
	for k in inst_labels.keys():
		if k in inst_idx:
			index = (row_idx, inst_idx[k])
			if index not in m.dict:
				m.dict[index] = 1.
			row_idx += 1
	return m

# F x N sparse matrix
def get_f_mask(inst_idx, feat_idx, feat_labels, documents):
	m = MySparseData()
	m.shape = [len(feat_labels), len(inst_idx)]
	for k, v in documents.items():
		if k in inst_idx:
			f_idx = 0
			doc_str = '_'.join(v)
			for f in sorted(feat_labels.keys()):
				if f in feat_idx:
					# determine if doc_str contains f
					if f in doc_str:
						index = (f_idx, inst_idx[k])
						if index not in m.dict:
							m.dict[index] = 1.
				f_idx += 1
	return m

# L x K 
def get_i_f_target(inst_idx, label_idx, inst_labels):
	m = np.zeros( (len(inst_labels), len(label_idx)) )
	row_idx = 0
	for k, l_dict in inst_labels.items():
		if k in inst_idx:
			for l, p in l_dict.items():
				if l in label_idx:
					m[ row_idx, label_idx[l] ] = p
			row_idx += 1
	return m

# (feat_tasks x N) x (feat_tasks x M) matrix
def get_g_doc_sparse(main_m, feat_idx, feat_tasks):
	m = MySparseData()
	num_inst = main_m.shape[0]
	num_feat = main_m.shape[1]
	num_task = len(feat_tasks)
	m.shape = [ num_task*num_inst, num_task*num_feat ]
	task_idx = 0
	for f in feat_tasks:
		if f in feat_idx:
			fid = feat_idx[f]
			for k, v in main_m.dict.items():
				i, j = k
				if j != fid: # mask out j-th dimension
					index = (task_idx*num_inst + i, task_idx*num_feat + j)
					m.dict[index] = v
			task_idx += 1
	return m

# (feat_tasks x N) x 1 matrix
def get_g_target(main_m, feat_idx, feat_tasks):
	num_inst = main_m.shape[0]
	num_task = len(feat_tasks)
	m = np.zeros( (num_inst * num_task, 1) )
	task_idx = 0
	for f in feat_tasks:
		if f in feat_idx:
			fid = feat_idx[f]
			for i, j in main_m.dict.keys():
				if j == fid:
					m[task_idx*i, 0] = 1.
			task_idx += 1
	return m

def write_obj_to_file(obj, path):
	f = codecs.open(path, 'w', encoding='utf-8')
	f.write(str(obj))
	f.close()

def read_obj_from_file(path):
	f = codecs.open(path, 'r', encoding='utf-8')
	t = ast.literal_eval(f.read().strip())
	f.close()
	return t

class InteractiveLearner(object):
	""" Interative learning classifier.

	Parameters
	----------
	is_sparse : bool, default: True
		Sparse or dense model. If sparse, then other params include pretrained_emb, encoder, and
		last_layer_size will be ignored.

	pretrained_emb : tuple, (word_list, ndarray), default: None
		The word_list contains M words. The ndarray is m x d size matrix.

	encoder : str, {'one_hot', 'tfidf', 'l2_norm', 'average', 'lstm'}, default: 'one_hot'
		It encodes the document as an M dimensional vector.
		'one_hot': the document is encoded as a one-hot vector.
		'tfidf': each word has tfidf weight.
		'l2_norm': the vector has unit l2 norm.
		'average': for dense model, the document is encoded as the average of word vectors.
		'lstm': for dense model, the document is encoded as the output of a LSTM.

	predictor : str, {'one_layer', 'two_layer'}, default: 'one_layer'
		It connects the output of encoder to the prediction targets.
		'one_layer': the predictor is just a softmax layer.
		'two_layer': the predictor is a two-layer neural network. Useful with feature subtasks.

	encoder_output_size : int, default: 100
		it is set to d (embedding size) if encoder = 'average'; otherwise it is the output size of LSTM layer.
	
	softmax_input_size : int, default: 100
		The size of the hidden layer before the last softmax.
		Useful if predictor = 'two_layer'
	 	
	Attributes
	----------
	

	"""
	def __init__(self, 
		is_sparse = True, 
		pretrained_emb = None, 
		encoder = 'one_hot', 
		predictor = 'one_layer',
		encoder_output_size = 100, 
		softmax_input_size = 100,

		session = tf.Session(),
		name = 'iLearner',
		learning_rate = 1e-2,
		num_epoch = 100,
		anneal_rate = 25,
		anneal_stop_epoch = 100,
		evaluation_interval = 10,
		batch_size = 32,
		random_state = 100
		):

		self.is_sparse = is_sparse
		self.pretrained_emb = pretrained_emb
		self.encoder_name = encoder
		self.predictor_name = predictor
		self.encoder_output_size = encoder_output_size
		self.softmax_input_size = softmax_input_size
		
		self.session = session
		self.name = name
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.anneal_rate = anneal_rate
		self.anneal_stop_epoch = anneal_stop_epoch
		self.evaluation_interval = evaluation_interval
		self.batch_size = batch_size
		self.random_state = random_state

		self._init = tf.random_normal_initializer(stddev=0.1, dtype=tf.float32)
		
# ================================================================================================
# training-related functions
# ================================================================================================

	def fit(self, documents, inst_labels, feat_labels, feat_tasks):
		""" Fit the model: encoder and predictor

		Parameters
		----------
		documents : a dictionary of (str, list) pairs. d[inst] = word_seq
			inst: str, instance id
			word_seq: a list of word strings

		inst_labels : dictionary. d[inst][label] = prob(label|inst)
			inst: str, instance id
			label: str, label name

		feat_labels : dictionary. d[feat][label] = prob(label|feat)
			feat: str, feature string
			label: str, label name

		feat_tasks : list. [f1, f2, ...]
			each element is a feature string.

		Returns
		------------
		self : object
			Returns self.

		"""
		# =========================================================================
		# extract input data

		self.feat_tasks = feat_tasks

		# print 'documents', documents

		if self.is_sparse:
			self.feat_idx = get_feature_sparse(documents, feat_labels, feat_tasks)
			self.inst_idx, self.doc = featurize_sparse(documents, self.feat_idx)
		else:
			pass

		# print 'self.inst_idx', self.inst_idx
		# print 'self.feat_idx', self.feat_idx
		# print 'self.doc', self.doc.dict, self.doc.shape

		self.label_idx = get_label_idx(inst_labels, feat_labels)
		# print 'self.label_idx', self.label_idx

		self.i_mask = get_i_mask(self.inst_idx, inst_labels)
		# print 'self.i_mask', self.i_mask.dict, self.i_mask.shape

		self.i_target = get_i_f_target(self.inst_idx, self.label_idx, inst_labels)
		# print 'self.i_target', self.i_target

		self.f_mask = get_f_mask(self.inst_idx, self.feat_idx, feat_labels, documents)
		# print 'self.f_mask', self.f_mask.dict, self.f_mask.shape

		self.f_target = get_i_f_target(self.feat_idx, self.label_idx, feat_labels)
		# print 'self.f_target', self.f_target
		
		if self.is_sparse:
			self.g_doc = get_g_doc_sparse(self.doc, self.feat_idx, feat_tasks)
		else:
			pass
		# print 'self.g_doc', self.g_doc.dict, self.g_doc.shape

		self.g_target = get_g_target(self.doc, self.feat_idx, feat_tasks)
		# print 'self.g_target', self.g_target

		# =========================================================================
		# computational graph for training, different than the model itself. It leads to the objective value.

		# no hidden layer
		self.softmax_input_size = self.doc.shape[1]

		self._build_inputs()
		self._build_variables()

		if self.is_sparse:
			X = self.ph_doc
			X_g = self.ph_g_doc
		else:
			pass

		# if there are hidden layers, append them to encoded data


		# ph_i_mask -> softmax
		logits = tf.sparse_tensor_dense_matmul(X, self.W)

		i_logits = tf.sparse_tensor_dense_matmul(self.ph_i_mask, logits)
		objective = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_i_target, logits=i_logits))


		# ph_f_mask -> softmax
		f_logits = tf.sparse_tensor_dense_matmul(self.ph_f_mask, logits)
		objective += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_f_target, logits=f_logits))

		# ph_g_mask -> softmax

		g_logits = tf.sparse_tensor_dense_matmul(X_g, self.W_g)
		objective += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ph_g_target, logits=g_logits))
		
		# build up objective: will use i_target, f_target, g_target

		loss_op = objective
		# =========================================================================
		# gradient pipeline
		self._opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) # seems to be better than Adam
		grads_and_vars = self._opt.compute_gradients(loss_op)
		train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

		# assign ops
		self.loss_op = loss_op
		self.train_op = train_op

		# training
		init_op = tf.global_variables_initializer()
		self.session.run(init_op)

		for t in range(1, self.num_epoch + 1):
			cost = self._batch_fit(self.doc, self.i_mask, self.i_target, self.f_mask, self.f_target, self.g_doc, self.g_target)
			
		print 'epoch', t, 'cost', cost

		# =========================================================================
		# save model
		self.model = self.session.run(self.W)


		# self.session.close()
		return self

	# ======= internal functions =======

	def _build_inputs(self):
		if self.is_sparse:
			# we need sparse place holders
			self.ph_doc	  = tf.sparse_placeholder(tf.float32, [None, None], 'doc')
			self.ph_i_mask   = tf.sparse_placeholder(tf.float32, [self.i_mask.shape[0], None], 'i_mask')
			self.ph_i_target = tf.placeholder(tf.float32, [None, len(self.label_idx)], 'i_target')
			self.ph_f_mask   = tf.sparse_placeholder(tf.float32, [self.f_mask.shape[0], None], 'f_mask')
			self.ph_f_target = tf.placeholder(tf.float32, [None, len(self.label_idx)], 'f_target')
			self.ph_g_doc	= tf.sparse_placeholder(tf.float32, [None, None], 'g_doc')
			self.ph_g_target = tf.placeholder(tf.float32, [None, 1], 'g_target')

		else:
			# we need dense place holders

			# self.ph_g_doc = tf.placeholder(tf.float32, [None, None, self.num_feat_subtasks], 'g_doc')
			# self.ph_g_target = tf.placeholder(tf.float32, [None, None, self.num_feat_subtasks], 'g_target')
			pass

	def _build_variables(self):

		self.W = tf.Variable(self._init([self.softmax_input_size, len(self.label_idx)]), name="W")
		self.W_g = tf.Variable(self._init([len(self.feat_tasks) * self.softmax_input_size, 1]), name="W_g")
			
	# model architecture is here. 'ph' = 'placeholder'
	# it is used only if there the encoder is parametric
	def _encoder(self, ph_doc):
		
		if 'average' in self.encoder_name:
			pass
		if 'lstm' in self.encoder_name:
			pass
		# encode docs as dense vectors

		return ph_doc

	def _batch_fit(self, doc, i_mask, i_target, f_mask, f_target, g_doc, g_target):
		if self.is_sparse:
			feed_dict = {self.ph_doc: tf.SparseTensorValue(indices=doc.dict.keys(), values=doc.dict.values(), dense_shape=doc.shape),
						 self.ph_i_mask: tf.SparseTensorValue(indices=i_mask.dict.keys(), values=i_mask.dict.values(), dense_shape=i_mask.shape),
						 self.ph_i_target: i_target,
						 self.ph_f_mask: tf.SparseTensorValue(indices=f_mask.dict.keys(), values=f_mask.dict.values(), dense_shape=f_mask.shape),
						 self.ph_f_target: f_target,
						 self.ph_g_doc: tf.SparseTensorValue(indices=g_doc.dict.keys(), values=g_doc.dict.values(), dense_shape=g_doc.shape),
						 self.ph_g_target: g_target }

		loss, _ = self.session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
		return loss

# ================================================================================================
# prediction-related functions
# ================================================================================================
	def predict(self, documents):
		""" Predict labels for each document

		Parameters
		----------
		documents : a dictionary of (str, list) pairs. d[inst] = word_seq
			inst: str, instance id
			word_seq: a list of word strings

		Returns
		------------
		A dictionary of predictions. d[inst][label] = prob(label|inst)
			inst: str, instance id
			label: str, label name
			prob(label|inst): float

		"""
		# the computation graph for prediction is different than that for training

		# predict_op = tf.argmax(main_task_logits, 1, name="predict_op")
		# predict_proba_op = tf.nn.softmax(main_task_logits, name="predict_proba_op")
		# predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")
		# feed_dict = {self._documents: docs}
		# return self.session.run(self.predict_op, feed_dict=feed_dict)
		
		inst_idx, doc = featurize_sparse(documents, self.feat_idx)
		
		X = tf.sparse_placeholder(tf.float32, [None, None])
		W = tf.placeholder(tf.float32, [None, None])
		logits = tf.sparse_tensor_dense_matmul(X, W)
		predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
		
		p = self.session.run(predict_proba_op, feed_dict={X: tf.SparseTensorValue(indices=doc.dict.keys(), values=doc.dict.values(), dense_shape=doc.shape), W: self.model})

		pred = {}
		for inst_id, i in inst_idx.items():
			pred[inst_id] = {}
			for label_id, j in self.label_idx.items():
				pred[inst_id][label_id] = p[i, j]
		return pred

	def explain(self, documents):
		""" Explain the prediction of each document

		Parameters
		----------
		documents : a dictionary of (str, list) pairs. d[inst] = word_seq
			inst: str, instance id
			word_seq: a list of word strings

		Returns
		------------
		A dictionary of explanations. d[inst][k][label] = val
			inst: str, instance id
			k: k-th word (starting from 0, may not have values for all words)
			label: str, label name
			val: float, importance of k-th word indicating a label

		"""
		val_threshold = np.amax(self.model) * 0.8

		exp = {}

		for k, v in documents.items():
			exp[k] = {}
			end = -1
			for feat in v:
				if feat in self.feat_idx:
					start = end + 1
					end = start + len(feat)
					span = (start, end, feat)
					exp[k][span] = {}
					i = self.feat_idx[feat]
					for label, j in self.label_idx.items():
						val = self.model[i, j]
						if val > val_threshold:
							exp[k][span][label] = val
		return exp

	# The following two functions may not be straightforward if each time the model can have different features.
	def load_model(self, model_dir):
		""" Load the model from a folder.
		A model includes all attributes of the class
		"""
		if not os.path.exists(model_dir):
			ValueError('learner::load_model(): model_dir does not exist: {}'.format(model_dir))
			
		main_param_path = os.path.join(model_dir, 'main_param.txt')
		m = np.loadtxt(main_param_path)
		if len(m.shape) == 1:
			m = m.reshape( (m.shape[0], 1) )
		self.model = m

		feat_map_path = os.path.join(model_dir, 'feat_idx.txt')
		self.feat_idx = read_obj_from_file(feat_map_path)

		label_map_path = os.path.join(model_dir, 'label_idx.txt')	
		self.label_idx = read_obj_from_file(label_map_path)

		return

	def save_model(self, model_dir):
		""" Save the model to a folder.
		A model includes all attributes of the class
		"""
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		main_param_path = os.path.join(model_dir, 'main_param.txt')
		np.savetxt(main_param_path, self.model)

		feat_map_path = os.path.join(model_dir, 'feat_idx.txt')
		write_obj_to_file(self.feat_idx, feat_map_path)
		
		label_map_path = os.path.join(model_dir, 'label_idx.txt')
		write_obj_to_file(self.label_idx, label_map_path)

		return

	


	

	

	



