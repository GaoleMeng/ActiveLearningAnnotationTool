
#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import sys
import os, codecs
import numpy as np
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # mute the CPU acceleration warnings
import tensorflow as tf

from collections import Counter
import scipy.sparse as sp

from constants import *
from evaluator import compute_metrics
from LSTMembedding import LSTMlogits

class MySparseData(object):
	def __init__(self):
		self.dict = {}
		self.shape = []

	def slice_row(self, slice_arr):
		new_idx = {}
		j = 0
		for i in sorted(slice_arr):
			new_idx[i] = j
			j += 1
		d = {}
		for i in slice_arr:
			for j in range(self.shape[1]):
				if (i, j) in self.dict:
					d[ (new_idx[i], j) ] = self.dict[(i, j)]
		m = MySparseData()
		m.dict = d
		m.shape = [len(slice_arr), self.shape[1]]
		return m

	def todense(self, dtype=np.float32):
		m = np.zeros( self.shape, dtype=dtype )
		for k, v in self.dict.items():
			i, j = k
			m[i, j] = v
		return m

def get_feature_idx(documents, max_vocab, feat_labels, feat_tasks):
	words = [i for l in documents.values() for i in l]
	word_freq = sorted(Counter(words).items(), key = lambda x: x[1], reverse = True)
	if max_vocab > 0:
		word_freq = word_freq[:max_vocab] # cut features
	feat_idx = dict(zip([e[0] for e in word_freq], range(len(word_freq)) ))
	
	# add other features
	feat_i = len(feat_idx)
	for k in feat_labels.keys() + feat_tasks:
		if k not in feat_idx:
			feat_idx[k] = feat_i
			feat_i += 1
	return feat_idx

def get_feature_stats(documents, feat_idx):
	count_vec = np.zeros(len(feat_idx))
	doc_freq_vec = np.zeros(len(feat_idx))

	for v in documents.values():
		uniq = {}
		for f in v:
			if f in feat_idx:
				fid = feat_idx[f]
				count_vec[fid] += 1.
				if fid not in uniq:
					uniq[fid] = 1
		for fid in uniq:
			doc_freq_vec[fid] += 1.
	return count_vec, doc_freq_vec

# tf can be:
#   'count': each dimension is a count
#   'binary': each dimension is binary
# if df is not None, each dimension is weighted by tfidf
def featurize_bow(documents, feat_idx, tf = 'count', df = None):
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
				elif value == 'count':
					m.dict[index] += 1.
	if idf != None:
		for i, j in m.dict:
			m.dict[(i, j)] = np.log(m.dict[(i, j)] + 1) * np.log(len(inst_idx) / (idf[j] + 1.) )

	return inst_idx, m

def featurize_bow2(documents, feat_idx, feat_df):
	# instead of returning a sparse matrix, return sp_ids and sp_weights
	pass

def featurize_seq(documents, feat_idx):
	inst_idx = dict(zip(sorted(documents.keys()),  range(len(documents)) ))
	seq_len_arr = [len(i) for i in documents.values()]
	seq_len = int(np.mean(seq_len_arr) + 2*np.std(seq_len_arr))
	# print '2 sigma seq_len:', seq_len
	seq_len = min(seq_len, max(seq_len_arr), MAX_TOKEN_SEQUENCE_LENGTH)
	m = np.zeros( (len(documents), seq_len), dtype=np.int32 )
	for k, v in documents.items():
		row_idx = inst_idx[k]
		for j, f in enumerate(v[:seq_len]):
			if f in feat_idx:
				token_id = feat_idx[f]
				m[row_idx, j] = token_id
	return inst_idx, m

def get_label_idx(inst_labels, feat_labels):
	label_idx = {}
	label_i = 0
	for l_dict in inst_labels.values() + feat_labels.values():
		for k, v in l_dict.items():
			if k not in label_idx:
				label_idx[k] = label_i
				label_i += 1
	return label_idx

# N x F sparse matrix: sim(f, x), feature-instance association (should be positive number)
def get_feat_inst_assn(inst_idx, feat_idx, feat_labels, documents):
	m = MySparseData()
	m.shape = [len(inst_idx), len(feat_labels)]
	for k, v in documents.items():
		if k in inst_idx:
			f_idx = 0
			# doc_str = '_'.join(v)
			v_dict = dict.fromkeys(v)
			for f in sorted(feat_labels.keys()):
				if f in feat_idx:
					# if f in doc_str:
					if f in v_dict:
						# determine the strength of association between f (feature) and v (document).
						# future work: make it non-binary (e.g. using embedding similarity)
						m.dict[(inst_idx[k], f_idx)] = 1.
				f_idx += 1
	return m

# N x K sparse matrix
def get_i_target(inst_idx, label_idx, inst_labels):
	m = MySparseData()
	m.shape = [len(inst_idx), len(label_idx)]
	for k, l_dict in inst_labels.items():
		if k in inst_idx:
			for l, p in l_dict.items():
				if l in label_idx:
					m.dict[ (inst_idx[k], label_idx[l]) ] = p
	return m

# F x K dense matrix
def get_f_target(feat_idx, label_idx, feat_labels):
	m = np.zeros( (len(feat_labels), len(label_idx)), dtype=np.float32 )
	row_idx = 0
	for f in sorted(feat_labels.keys()):
		if f in feat_idx:
			for l, p in feat_labels[f].items():
				if l in label_idx:
					m[ row_idx, label_idx[l] ] = p
			row_idx += 1
	return m

# [This function should not be used, as we don't need MTL with BOW features]
# (feat_tasks x N) x (feat_tasks x M) matrix
def get_g_doc_bow(main_m, feat_idx, feat_tasks):
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

# (feat_tasks x N x l) tensor
def get_g_doc_seq(main_m, feat_idx, feat_tasks):
	num_inst = main_m.shape[0]
	seq_len = main_m.shape[1]
	num_task = len(feat_tasks)
	m = np.zeros( (num_task, num_inst, seq_len), dtype=np.int32 )
	task_idx = 0
	for f in feat_tasks:
		if f in feat_idx:
			token_id = feat_idx[f]
			for i in range(num_inst):
				for j in range(seq_len):
					if main_m[i, j] != token_id: # skip the token
						m[task_idx, i, j] = main_m[i, j]
			task_idx += 1
	return m

# [This function should not be used, as we don't need MTL with BOW features]
# (feat_tasks x N) x 1 matrix
def get_g_target_bow(main_m, feat_idx, feat_tasks):
	num_inst = main_m.shape[0]
	num_task = len(feat_tasks)
	m = np.zeros( (num_inst * num_task, 1) )
	task_idx = 0
	for f in feat_tasks:
		if f in feat_idx:
			fid = feat_idx[f]
			for i, j in main_m.dict.keys():
				if j == fid:
					m[task_idx*num_inst + i, 0] = 1.
			task_idx += 1
	return m

# feat_tasks x N x 1 tensor
def get_g_target_seq(main_m, feat_idx, feat_tasks):
	num_inst = num_inst = main_m.shape[0]
	seq_len = main_m.shape[1]
	num_task = len(feat_tasks)
	m = np.zeros( (num_task, num_inst, 1) )
	task_idx = 0
	for f in feat_tasks:
		if f in feat_idx:
			token_id = feat_idx[f]
			for i in range(num_inst):
				for j in range(seq_len):
					if main_m[i, j] == token_id:
						m[task_idx, i, 0] = 1.
						break
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

def prepare_batch_intervals(n_train, batch_size):
	batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
	if len(batches) == 0:
		batches = [(0, n_train)]
	elif batches[-1][1] != n_train:
		batches += [(batches[-1][1], n_train)]
	return batches

def get_info_gain_from_predicted_labels(my_sparse, raw_pred):
	row_idx = [i for i, j in my_sparse.dict.keys()]
	col_idx = [j for i, j in my_sparse.dict.keys()]
	# print 'max(row_idx)', max(row_idx)
	# print 'max(col_idx)', max(col_idx)
	# print 'my_sparse.shape', my_sparse.shape
	X_ind = sp.csc_matrix( ([1.] * len(my_sparse.dict), (row_idx, col_idx)), shape=my_sparse.shape )
	n_doc = X_ind.shape[0]
	feature_count = X_ind.sum(axis=0).transpose() + 1e-100 # V x 1
	label_count = raw_pred.sum(axis=0) + 1e-100 # C x 1
	feature_label_cooc = X_ind.transpose().dot(raw_pred) # V x C
	label_given_feature = feature_label_cooc / feature_count
	label_given_no_feature = (label_count - feature_label_cooc) / ( n_doc-feature_count + 1e-50 )
	label_prob = label_count / label_count.sum()
	feature_prob = feature_count / n_doc

	# info gain = H(c) - H(c|f)
	info_gain = np.multiply(feature_prob, np.multiply(label_given_feature, np.log(label_given_feature + 1e-50)).sum(axis=1)) # sum over labels
	info_gain += np.multiply(1-feature_prob, np.multiply(label_given_no_feature, np.log(label_given_no_feature + 1e-50)).sum(axis=1))
	info_gain -= np.multiply(label_prob,np.log(label_prob + 1e-50) ).sum()
	return info_gain, label_given_feature

# dense architecture definition strings
class DenseArch(object):
	ONE_LAYER = 'one_layer'
	TWO_LAYER_AVG_EMB = 'two_layer:avg_emb'
	TWO_LAYER_TFIDF_EMB = 'two_layer:tfidf_emb'
	TWO_LAYER_LSTM = 'two_layer:LSTM'

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

	embedding_size : int, default: 100
		it is set to d (embedding size) if encoder = 'average'; otherwise it is the output size of LSTM layer.
	 	
	Attributes
	----------
	

	"""
	def __init__(self, 
		is_sparse = True, 
		pretrained_emb = None,
		embedding_size = 100, 
		dense_architecture = DenseArch.ONE_LAYER,
		max_vocab = -1,

		session = tf.Session(),
		name = 'iLearner',
		learning_rate = 1e-2,
		num_epoch = 100,
		batch_size = 32,
		validation_interval = 5,
		random_state = 100,
		pre_trained_embedding = False
		):

		self.is_sparse = is_sparse
		self.pretrained_emb = pretrained_emb
		self.embedding_size = embedding_size
		self.dense_architecture = dense_architecture
		self.max_vocab = max_vocab
		
		self.session = session
		self.name = name
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.batch_size = batch_size
		self.validation_interval = validation_interval
		self.random_state = random_state
		self.pre_trained_embedding = pre_trained_embedding

		self._init = tf.random_normal_initializer(stddev=0.1, dtype=tf.float32)

		self.tmp_lstm_model = LSTMlogits();

		
# ================================================================================================
# training-related functions
# ================================================================================================

	def fit(self, documents, inst_labels, feat_labels, feat_tasks, model_dir, validation_set = None):
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
		if len(inst_labels) == 0 and len(feat_labels) == 0:
			sys.stderr.write('learner::fit(): neither instance label nor feature label is provided. Nothing to fit.\n')
			return self
		# =========================================================================
		# extract input data
		# make meta data as member variables, so that they can be accessed from outside
		self.inst_labels = inst_labels
		self.feat_labels = feat_labels
		self.feat_tasks = feat_tasks

		
		# print 'documents', documents

		sys.stderr.write('learner::fit(): Extracting features ...\n')
		self.feat_idx = get_feature_idx(documents, self.max_vocab, feat_labels, feat_tasks)
		_, self.feat_doc_freq = get_feature_stats(documents, self.feat_idx)

		if self.is_sparse:
			self.inst_idx, doc = featurize_bow(documents, self.feat_idx, tf = 'count', idf = self.feat_doc_freq)
		elif self.dense_architecture == DenseArch.TWO_LAYER_TFIDF_EMB:
			self.inst_idx, doc = featurize_seq_sp(documents, self.feat_idx)
		else:
			self.feat_idx = dict([('PADDING_TOKEN',0)] + [(k, v + 1) for k, v in self.feat_idx.items()])
			self.inst_idx, doc = featurize_seq(documents, self.feat_idx)

		# print 'self.inst_idx', self.inst_idx
		# print 'self.feat_idx', self.feat_idx
		# print 'doc', doc.dict, doc.shape
		# print 'doc', doc, doc.shape

		self.label_idx = get_label_idx(inst_labels, feat_labels)
		# print 'self.label_idx', self.label_idx

		i_target = get_i_target(self.inst_idx, self.label_idx, inst_labels)
		# print 'i_target', i_target

		feat_doc_assn = get_feat_inst_assn(self.inst_idx, self.feat_idx, feat_labels, documents)
		# print 'feat_doc_assn', feat_doc_assn.dict, feat_doc_assn.shape
		# print 'feat_doc_assn', feat_doc_assn.shape

		f_target = get_f_target(self.feat_idx, self.label_idx, feat_labels)
		# print 'feat_labels', feat_labels
		# print 'f_target', f_target
		
		if self.is_sparse:
			g_doc = None
			g_target = None
			# g_doc = get_g_doc_bow(doc, self.feat_idx, feat_tasks)
			# g_target = get_g_target_bow(doc, self.feat_idx, feat_tasks)
			# print 'g_doc', g_doc.dict, g_doc.shape
			pass
		else:
			g_doc = get_g_doc_seq(doc, self.feat_idx, feat_tasks)
			g_target = get_g_target_seq(doc, self.feat_idx, feat_tasks)
			# print 'g_doc', g_doc, g_doc.shape
			# print 'g_target', g_target

		if self.dense_architecture == DenseArch.ONE_LAYER:
			self.embedding_size = len(self.label_idx)

		if validation_set != None:
			val_documents, val_labels = validation_set
			val_max_perf = -1.0

		# =========================================================================
		# computational graph for training, different than the model itself. It leads to the objective value.

		sys.stderr.write('learner::fit(): Building computational graph ...\n')
		# no hidden layer

		self._build_inputs()
		self._build_variables()

		objective = tf.constant(0, dtype=tf.float32)

		# build up objective: will use i_target, f_target, g_target
		if self.is_sparse:
			if len(inst_labels) > 0:
				i_logits = tf.sparse_tensor_dense_matmul(self.ph_i_doc, self.W)
				i_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_i_target, logits=i_logits))
				objective += tf.cond(self.ph_i_dotrain, lambda: i_loss, lambda: 0.)

			if len(feat_labels) > 0:
				f_logits = tf.sparse_tensor_dense_matmul(self.ph_f_doc, self.W)
				f_pred = self._feature_label_dist(f_logits, self.ph_feat_doc_assn)
				f_loss = -tf.reduce_sum( self.ph_f_target * tf.log(f_pred + 1e-10) ) # cross entropy
				objective += tf.cond(self.ph_f_dotrain, lambda: f_loss, lambda: 0.)

		else: # prepend intermediate/encoding module before softmax. Now it's just an embedding lookup. Can add LSTM.
			if len(inst_labels) > 0:
				i_logits = self._dense_arch(self.W_emb, self.W, self.ph_i_doc)
				
				i_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_i_target, logits=i_logits))
				objective += tf.cond(self.ph_i_dotrain, lambda: i_loss, lambda: 0.)

			if len(feat_labels) > 0:
				f_logits = self._dense_arch(self.W_emb, self.W, self.ph_f_doc)
				f_pred = self._feature_label_dist(f_logits, self.ph_feat_doc_assn)
				f_loss = -tf.reduce_sum( self.ph_f_target * tf.log(f_pred + 1e-10) )
				objective += tf.cond(self.ph_f_dotrain, lambda: f_loss, lambda: 0.)

			if len(feat_tasks) > 0:
				if self.dense_architecture != DenseArch.ONE_LAYER:
					g_logits = self._dense_arch(self.W_emb, self.W_g, self.ph_g_doc, reduce_dim=2) # [g x N x l x k] => [g x N x k]
					objective += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ph_g_target, logits=g_logits))

		loss_op = objective
		# =========================================================================
		# gradient pipeline
		self._opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate) # seems to be better than Adam
		# self._opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) # seems to be better than Adam
		grads_and_vars = self._opt.compute_gradients(loss_op)
		train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

		# assign ops
		self.loss_op = loss_op
		self.train_op = train_op

		# training
		init_op = tf.global_variables_initializer()
		self.session.run(init_op)

		sys.stderr.write('learner::fit(): Performing gradient descend ...\n')

		# =========================================================================
		# perform mini-batch SGD training
		batches = prepare_batch_intervals(len(self.inst_idx), self.batch_size)
		val_perf_drop_counter = 0
		for t in range(1, self.num_epoch + 1):
			np.random.shuffle(batches)
			total_cost = 0.0
			i = 1
			for start, end in batches:
				cost = self._batch_fit(range(start, end), doc, i_target, feat_doc_assn, f_target, g_doc, g_target)
				total_cost += cost
				# print 'batch', i, i*self.batch_size, cost
				i += 1
			print 'epoch', t, 'total_cost', total_cost
			if t % self.validation_interval == 0 and validation_set != None:
				self._refresh_model_params()
				val_max_perf, is_lower = self._eval_validation_set(val_documents, val_labels, val_max_perf, model_dir)
				if is_lower:
					val_perf_drop_counter += 1
				else: # clear counter
					val_perf_drop_counter = 0
				if val_perf_drop_counter >= STOP_AFTER_VAL_PERF_DROP_COUNT:
					break

		print 'epoch', t, 'total_cost', total_cost

		# =========================================================================
		if validation_set != None:
			self._refresh_model_params()
			self._eval_validation_set(val_documents, val_labels, val_max_perf, model_dir)
		else: # no validation set: save the final model
			self._refresh_model_params()
			self.save_model(model_dir)
		
		return self

	# ======= internal functions =======

	def _build_inputs(self):
		self.ph_feat_doc_assn = tf.placeholder(tf.float32, [None, None], 'ph_feat_doc_assn')
		self.ph_i_target = tf.placeholder(tf.float32, [None, len(self.label_idx)], 'i_target')
		self.ph_f_target = tf.placeholder(tf.float32, [None, len(self.label_idx)], 'f_target')

		self.ph_i_dotrain = tf.placeholder(tf.bool, [])
		self.ph_f_dotrain = tf.placeholder(tf.bool, [])

		if self.is_sparse:
			# we need sparse place holders
			self.ph_i_doc = tf.sparse_placeholder(tf.float32, [None, None], 'i_doc')
			self.ph_f_doc = tf.sparse_placeholder(tf.float32, [None, None], 'f_doc')
		else:
			# we need dense place holders
			self.ph_i_doc = tf.placeholder(tf.int32, [None, None], 'i_doc')
			self.ph_f_doc = tf.placeholder(tf.int32, [None, None], 'f_doc')
			self.ph_g_doc = tf.placeholder(tf.int32, [None, None, None], 'g_doc')
			self.ph_g_target = tf.placeholder(tf.float32, [None, None, 1], 'g_target')

	def _build_variables(self):
		if self.is_sparse:
			self.W = tf.Variable(self._init([len(self.feat_idx), len(self.label_idx)], dtype=tf.float32), name="W")
			# self.W_g = tf.Variable(self._init([len(self.feat_tasks) * len(self.feat_idx), 1], dtype=tf.float32), name="W_g")

		else:
			nil_word_slot = tf.zeros([1, self.embedding_size], dtype=tf.float32) #### zero padding variable here ...
			A = tf.concat([ nil_word_slot, self._init([len(self.feat_idx), self.embedding_size], dtype=tf.float32) ], 0) # embedding matrix: V x k
			if self.pretrained_emb is not None:
				#A = tf.concat([ nil_word_slot, self._init([len(self.feat_idx), self.embedding_size], dtype=tf.float32) ], 0)
				#A = np.random.normal([len(self.feat_idx)+1, self.embedding_size], dtype=np.float32);
				A = np.random.normal(scale = 0.1, size = (len(self.feat_idx)+1, self.embedding_size)).astype(np.float32);
				A[0,:] = np.zeros([1, self.embedding_size]);
				for k,v in self.pretrained_emb.items():
					if (k in self.feat_idx):
						A[self.feat_idx[k]] = np.array(v);
			self.W_emb = tf.Variable(A, name="W_emb")

			self.W = tf.Variable(self._init([self.embedding_size, len(self.label_idx)], dtype=tf.float32), name="W")
			self.W_g = tf.Variable(self._init([len(self.feat_tasks), self.embedding_size, 1], dtype=tf.float32), name="W_g")

	# dense model architecture is here.
	# token sequence, params -> logits before computing the final loss
	# Note that the params can be of different types:
	# when used in training: W_emb, W are variables
	# when used in prediction, W_emb, W are placeholders
	def _dense_arch(self, W_emb, W, doc, weights = None, reduce_dim = 1):
		if self.dense_architecture == DenseArch.ONE_LAYER:
			logits = tf.reduce_mean(tf.nn.embedding_lookup(W_emb, doc), reduce_dim) # [N x l x k] => [N x k]
		elif self.dense_architecture == DenseArch.TWO_LAYER_AVG_EMB:
			X = tf.sigmoid(tf.reduce_mean(tf.nn.embedding_lookup(W_emb, doc), reduce_dim)) # [N x l x k] => [N x k]
			logits = tf.matmul(X, W)
		elif self.dense_architecture == DenseArch.TWO_LAYER_LSTM:
			logits = tf.matmul(self.tmp_lstm_model.get_embedding(doc, W_emb), W);
		return logits

	def _feature_label_dist(self, logits, feat_doc_assn):
		f_pred = tf.matmul( tf.transpose(feat_doc_assn), tf.nn.softmax(logits) )
		f_pred = f_pred / (tf.reduce_sum(f_pred, 1, keep_dims=True) + 1e-50) # normalize such that each row sums to 1
		return f_pred

	def _batch_fit(self, batch_indices, doc, i_target, feat_doc_assn, f_target, g_doc, g_target):
		feed_dict = {}
		if self.is_sparse:
			batch_i_target = i_target.slice_row(batch_indices)
			if len(batch_i_target.dict) > 0: # there exists non-empty rows (valid targets) in the batch
				# print 'batch_i_target > 0'
				ne_idx = np.unique([i for i, j in batch_i_target.dict.keys()])
				good_batch_indices = [batch_indices[i] for i in ne_idx]
				i_doc = doc.slice_row(good_batch_indices)
				feed_dict[self.ph_i_doc] = tf.SparseTensorValue(indices=i_doc.dict.keys(), values=i_doc.dict.values(), dense_shape=i_doc.shape)
				feed_dict[self.ph_i_target] = i_target.slice_row(good_batch_indices).todense()
				feed_dict[self.ph_i_dotrain] = True
			else:
				# print 'batch_i_target == 0'
				feed_dict[self.ph_i_doc] = tf.SparseTensorValue(indices=[(0,0)], values=[0.], dense_shape=[1,doc.shape[1]])
				feed_dict[self.ph_i_target] = np.ones( (1,i_target.shape[1]) )
				feed_dict[self.ph_i_dotrain] = False

			batch_fd_assn = feat_doc_assn.slice_row(batch_indices)
			if len(batch_fd_assn.dict) > 0:
				# print 'batch_fd_assn > 0'
				ne_idx = np.unique([i for i, j in batch_fd_assn.dict.keys()])
				# print 'ne_idx', ne_idx
				good_batch_indices = [batch_indices[i] for i in ne_idx]
				# print 'good_batch_indices', good_batch_indices
				f_doc = doc.slice_row(good_batch_indices)
				# print 'f_doc.dict', f_doc.dict
				# print 'f_doc.shape', f_doc.shape
				# print 'feat_doc_assn', feat_doc_assn.slice_row(good_batch_indices).todense()
				good_batch_fd_assn = feat_doc_assn.slice_row(good_batch_indices)
				relevant_feat = np.unique([j for i, j in good_batch_fd_assn.dict.keys()])
				feed_dict[self.ph_f_doc] = tf.SparseTensorValue(indices=f_doc.dict.keys(), values=f_doc.dict.values(), dense_shape=f_doc.shape)
				feed_dict[self.ph_f_target] = f_target[relevant_feat,:]
				feed_dict[self.ph_feat_doc_assn] = good_batch_fd_assn.todense()[:,relevant_feat]
				feed_dict[self.ph_f_dotrain] = True
			else:
				# print 'batch_fd_assn == 0'
				feed_dict[self.ph_f_doc] = tf.SparseTensorValue(indices=[(0,0)], values=[0.], dense_shape=[1,doc.shape[1]])
				feed_dict[self.ph_f_target] = np.ones( (1,f_target.shape[1]) )
				feed_dict[self.ph_feat_doc_assn] = np.ones( (1,1) )
				feed_dict[self.ph_f_dotrain] = False

		else:
			batch_i_target = i_target.slice_row(batch_indices)
			if len(batch_i_target.dict) > 0:
				ne_idx = np.unique([i for i, j in batch_i_target.dict.keys()])
				good_batch_indices = [batch_indices[i] for i in ne_idx]
				feed_dict[self.ph_i_doc] = doc[good_batch_indices, ]
				feed_dict[self.ph_i_target] = i_target.slice_row(good_batch_indices).todense()
				feed_dict[self.ph_i_dotrain] = True
			else:
				feed_dict[self.ph_i_doc] = doc[[0], :]
				feed_dict[self.ph_i_target] = np.ones( (1,i_target.shape[1]) )
				feed_dict[self.ph_i_dotrain] = False

			batch_fd_assn = feat_doc_assn.slice_row(batch_indices)
			if len(batch_fd_assn.dict) > 0:
				ne_idx = np.unique([i for i, j in batch_fd_assn.dict.keys()])
				good_batch_indices = [batch_indices[i] for i in ne_idx]
				good_batch_fd_assn = feat_doc_assn.slice_row(good_batch_indices)
				relevant_feat = np.unique([j for i, j in good_batch_fd_assn.dict.keys()])
				feed_dict[self.ph_f_doc] = doc[good_batch_indices, ]
				feed_dict[self.ph_f_target] = f_target[relevant_feat,:]
				feed_dict[self.ph_feat_doc_assn] = good_batch_fd_assn.todense()[:,relevant_feat]
				feed_dict[self.ph_f_dotrain] = True
			else:
				feed_dict[self.ph_f_doc] = doc[[0], :]
				feed_dict[self.ph_f_target] = np.ones( (1,f_target.shape[1]) )
				feed_dict[self.ph_feat_doc_assn] = np.ones( (1,1) )
				feed_dict[self.ph_f_dotrain] = False


			if g_target.shape[0] > 0:
				feed_dict[self.ph_g_doc] = g_doc[:, batch_indices, :]
				feed_dict[self.ph_g_target] = g_target[:, batch_indices, :]

		loss, _ = self.session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
		return loss

	def _refresh_model_params(self):
		self.param_W = self.session.run(self.W)
		if not self.is_sparse:
			self.param_W_emb = self.session.run(self.W_emb)

	def _eval_validation_set(self, documents, labels, max_perf, model_dir):
		preds = self.predict(documents)
		m = compute_metrics(labels, preds)
		print 'validation perf.:', \
			'accu={:.4f}'.format(m['accuracy']), \
			'macro_p={:.4f}'.format(m['macro_avg_precision']), \
			'macro_r={:.4f}'.format(m['macro_avg_recall'])
		if m['accuracy'] >= max_perf:
			max_perf = m['accuracy'] # update max perf
			self.save_model(model_dir)
			is_lower = False
		else:
			is_lower = True
		return max_perf, is_lower

# ================================================================================================
# prediction-related functions
# ================================================================================================
	def predict(self, documents, raw = False):
		""" Predict labels for each document

		Parameters
		----------
		documents : a dictionary of (str, list) pairs. d[inst] = word_seq
			inst: str, instance id
			word_seq: a list of word strings
			raw: bool, raw=True means output raw predicted matrix. raw=False output cooked dictionary

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
		
		# sys.stderr.write('learner::predict(): performing prediction ...\n')

		if self.is_sparse:
			inst_idx, doc = featurize_bow(documents, self.feat_idx, tf = 'count', df = self.feat_doc_freq)
			
			X = tf.sparse_placeholder(tf.float32, [None, None])
			W = tf.placeholder(tf.float32, [None, None])
			logits = tf.sparse_tensor_dense_matmul(X, W)
			predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

			p = self.session.run(predict_proba_op, feed_dict={X: tf.SparseTensorValue(indices=doc.dict.keys(), values=doc.dict.values(), dense_shape=doc.shape), W: self.param_W})
		else:
			inst_idx, doc = featurize_seq(documents, self.feat_idx)

			X = tf.placeholder(tf.int32, [None, None])
			W_emb = tf.placeholder(tf.float32, (len(self.feat_idx) + 1, self.embedding_size))
			W = tf.placeholder(tf.float32, [None, None])
			logits = self._dense_arch(W_emb, W, X)			
			predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

			p = self.session.run(predict_proba_op, feed_dict={X: doc, W_emb: self.param_W_emb, W: self.param_W})

		if raw:
			return p

		pred = {}
		for inst_id, i in inst_idx.items():
			pred[inst_id] = {}
			for label_id, j in self.label_idx.items():
				pred[inst_id][label_id] = p[i, j]
		return pred

	def explain(self, auxiliary_documents, documents_to_explain):
		""" Explain the prediction of a set of documents, given a set of auxiliary documents.

		Parameters
		----------
		auxiliary_documents: a dictionary of (str, list) pairs. d[inst] = word_seq
			inst: str, instance id
			word_seq: a list of word strings

		documents_to_explain : the same format as auxiliary_documents
			
		Returns
		------------
		A dictionary of explanations. d[inst][span][label] = val
			inst: str, instance id
			span: a tuple of (start, end, feat). doc[start:end] = feat
			label: str, label name
			val: a score for the label
		"""
		# sys.stderr.write('learner::explain(): performing explanation ...\n')

		# strategy: explanation by feature ranking
		# 1) generate predicted (soft) labels for auxiliary documents using current model;
		# 2) perform info-gain feature selection: a map from feature to label
		# 3) explanation = features with high info gain.

		aux_pred = self.predict(auxiliary_documents, raw = True)
		_, aux_doc = featurize_bow(auxiliary_documents, self.feat_idx)
		info_gain, label_given_feat = get_info_gain_from_predicted_labels(aux_doc, aux_pred)

		top_feat_idx = [k for k, v in sorted(enumerate(info_gain), key = lambda x: x[1], reverse = True)]
		reverse_feat_idx = dict([(v, k) for k, v in self.feat_idx.items()])
		top_feat_label = {}
		for l, l_idx in self.label_idx.items():
			num_feat_for_l = 0
			for i in top_feat_idx:
				feat = reverse_feat_idx[i]
				if feat not in EXP_STOPWORDS:
					if l_idx == np.argmax(label_given_feat[i, :]):
						top_feat_label[ feat ] = l
						num_feat_for_l += 1
						if num_feat_for_l == 50:
							break

		exp_pred = self.predict(documents_to_explain, raw = False)
		exp = {}
		for k, v in documents_to_explain.items():
			pred_lbl = sorted(exp_pred[k].items(), key = lambda x: x[1], reverse = True)[0][0]
			exp[k] = {}
			end = -1
			for feat in v:
				start = end + 1
				end = start + len(feat)
				if feat in top_feat_label:
					feat_lbl = top_feat_label[feat]
					if feat_lbl == pred_lbl:
						span = (start, end, feat)
						exp[k][span] = { feat_lbl : info_gain[self.feat_idx[feat], 0] }
		# print exp
		return exp

	# The following two functions may not be straightforward if each time the model can have different features.
	def load_model(self, model_dir):
		""" Load the model from a folder.
		A model includes all attributes of the class
		"""
		if not os.path.exists(model_dir):
			ValueError('learner::load_model(): model_dir does not exist: {}'.format(model_dir))

		hyperparam_path = os.path.join(model_dir, 'hyperparam.txt')
		hp = read_obj_from_file(hyperparam_path)
		self.is_sparse = hp['is_sparse']
		self.embedding_size = hp['embedding_size']
		self.dense_architecture = hp['dense_architecture']
		self.max_vocab = hp['max_vocab']

		param_W_path = os.path.join(model_dir, 'param_W.txt')
		m = np.loadtxt(param_W_path)
		if len(m.shape) == 1:
			m = m.reshape( (m.shape[0], 1) )
		self.param_W = m

		if not self.is_sparse:
			param_W_emb_path = os.path.join(model_dir, 'param_W_emb.txt')
			self.param_W_emb = np.loadtxt(param_W_emb_path)

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

		hyperparam = {'is_sparse': self.is_sparse
					 ,'embedding_size': self.embedding_size
					 ,'dense_architecture': self.dense_architecture
					 ,'max_vocab': self.max_vocab
					}
		hyperparam_path = os.path.join(model_dir, 'hyperparam.txt')
		write_obj_to_file(hyperparam, hyperparam_path)

		param_W_path = os.path.join(model_dir, 'param_W.txt')
		np.savetxt(param_W_path, self.param_W)

		if not self.is_sparse:
			param_W_emb_path = os.path.join(model_dir, 'param_W_emb.txt')
			np.savetxt(param_W_emb_path, self.param_W_emb)

		feat_map_path = os.path.join(model_dir, 'feat_idx.txt')
		write_obj_to_file(self.feat_idx, feat_map_path)
		
		label_map_path = os.path.join(model_dir, 'label_idx.txt')
		write_obj_to_file(self.label_idx, label_map_path)

		return

# ===============================================================================================
# Multinomial Naive Bayes classifier
# Can be trained by labeled instances and features.
# Sharing the interface of InteractiveLearner 

class InteractiveLearnerNaiveBayes(InteractiveLearner):
	def __init__(self,
		max_vocab = -1,
		laplacian_alpha = 1. ,
		feat_label_pseudo_count = 10.,

		session = tf.Session()
		):

		self.is_sparse = True
		self.max_vocab = max_vocab
		self.laplacian_alpha = laplacian_alpha
		self.feat_label_pseudo_count = feat_label_pseudo_count

		self.session = session

	def fit(self, documents, inst_labels, feat_labels, feat_tasks, model_dir, validation_set = None):
		if len(inst_labels) == 0 and len(feat_labels) == 0:
			sys.stderr.write('InteractiveLearnerNaiveBayes::fit(): neither instance label nor feature label is provided. Nothing to fit.\n')
			return self
		sys.stderr.write('InteractiveLearnerNaiveBayes::fit() ...\n')
		# =========================================================================
		# extract input data
		self.feat_idx = get_feature_idx(documents, self.max_vocab, feat_labels, feat_tasks)
		self.inst_idx, doc = featurize_bow(documents, self.feat_idx)
		self.label_idx = get_label_idx(inst_labels, feat_labels)
		i_target = get_i_target(self.inst_idx, self.label_idx, inst_labels)
		f_target = get_f_target(self.feat_idx, self.label_idx, feat_labels)

		# =========================================================================
		# estimate nb model (TODO: add a bias term)

		# laplacian prior
		self.param_W = np.ones( (len(self.feat_idx), len(self.label_idx)) ) * self.laplacian_alpha
		# self.param_W = np.zeros( (len(self.feat_idx), len(self.label_idx)) )

		# print 'self.param_W', self.param_W
		self.doc_freq = np.zeros( (1, doc.shape[1]) )
		for i, j in doc.dict.keys():
			self.doc_freq[0, j] += 1.

		# accumulate counts
		if len(i_target.dict) > 0:
			row_idx = [i for i, j in doc.dict.keys()]
			col_idx = [j for i, j in doc.dict.keys()]
			X = sp.csr_matrix( (doc.dict.values(), (col_idx, row_idx)), shape=[doc.shape[1], doc.shape[0]] ) # count
			# X = sp.csr_matrix( ([1.]*len(doc.dict), (col_idx, row_idx)), shape=[doc.shape[1], doc.shape[0]] ) # binary

			# TFIDF transformation
			X = X.log1p().multiply( np.log( doc.shape[0] / (self.doc_freq.transpose()+1.) ) )

			self.param_W += X.dot(i_target.todense())

		# print 'self.param_W', self.param_W

		# accumulate pseudo counts
		row_idx = 0
		for f in sorted(feat_labels.keys()):
			if f in self.feat_idx:
				self.param_W[self.feat_idx[f], :] += f_target[row_idx, :] * self.feat_label_pseudo_count
				row_idx += 1

		# print 'self.param_W', self.param_W

		# normalize by column, take log
		col_sum = self.param_W.sum(axis=0)
		self.param_W = np.log(self.param_W) - np.log(col_sum[np.newaxis, :])

		# print 'self.param_W', self.param_W

		col_mean = np.mean(self.param_W, axis=0)
		self.param_W = self.param_W - col_mean[np.newaxis, :]

		# print 'self.param_W', self.param_W

		# preds = self.predict(documents)
		# m = compute_metrics(inst_labels, preds)
		# print 'validation perf.:', \
		# 	'accu={:.4f}'.format(m['accuracy']), \
		# 	'macro_p={:.4f}'.format(m['macro_avg_precision']), \
		# 	'macro_r={:.4f}'.format(m['macro_avg_recall'])

		self.save_model(model_dir)
		return self

	def predict(self, documents, raw = False):
		inst_idx, doc = featurize_bow(documents, self.feat_idx)

		row_idx = [i for i, j in doc.dict.keys()]
		col_idx = [j for i, j in doc.dict.keys()]
		X_doc = sp.csr_matrix( (doc.dict.values(), (row_idx, col_idx)), shape=doc.shape ) # count
		# X_doc = sp.csr_matrix( ([1.]*len(doc.dict), (row_idx, col_idx)), shape=doc.shape ) # binary

		# TFIDF transformation
		X_doc = X_doc.log1p().multiply( np.log( doc.shape[0] / (self.doc_freq+1.) ) )

		p = X_doc.dot(self.param_W)

		max_col = np.max(p, axis=1)
		p = p - max_col[:, np.newaxis]
		p = np.exp(p)
		sum_col = np.sum(p, axis=1)
		p = p / sum_col[:, np.newaxis]

		if raw:
			return p

		pred = {}
		for inst_id, i in inst_idx.items():
			pred[inst_id] = {}
			for label_id, j in self.label_idx.items():
				pred[inst_id][label_id] = p[i, j]
		return pred

	def explain(self, auxiliary_documents, documents_to_explain):
		return super(InteractiveLearnerNaiveBayes, self).explain(auxiliary_documents, documents_to_explain)


	def save_model(self, model_dir):
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		hyperparam = {'is_sparse': True}
		hyperparam_path = os.path.join(model_dir, 'hyperparam.txt')
		write_obj_to_file(hyperparam, hyperparam_path)

		doc_freq_path = os.path.join(model_dir, 'doc_freq.txt')
		np.savetxt(doc_freq_path, self.doc_freq)

		param_W_path = os.path.join(model_dir, 'param_W.txt')
		np.savetxt(param_W_path, self.param_W)

		feat_map_path = os.path.join(model_dir, 'feat_idx.txt')
		write_obj_to_file(self.feat_idx, feat_map_path)
		
		label_map_path = os.path.join(model_dir, 'label_idx.txt')
		write_obj_to_file(self.label_idx, label_map_path)

	def load_model(self, model_dir):
		""" Load the model from a folder.
		A model includes all attributes of the class
		"""
		if not os.path.exists(model_dir):
			ValueError('learner::load_model(): model_dir does not exist: {}'.format(model_dir))

		hyperparam_path = os.path.join(model_dir, 'hyperparam.txt')
		hp = read_obj_from_file(hyperparam_path)
		self.is_sparse = hp['is_sparse']

		param_W_path = os.path.join(model_dir, 'param_W.txt')
		m = np.loadtxt(param_W_path)
		if len(m.shape) == 1:
			m = m.reshape( (m.shape[0], 1) )
		self.param_W = m

		doc_freq_path = os.path.join(model_dir, 'doc_freq.txt')
		m = np.loadtxt(doc_freq_path)
		if len(m.shape) == 1:
			m = m.reshape( (1, m.shape[0]) )
		self.doc_freq = m

		feat_map_path = os.path.join(model_dir, 'feat_idx.txt')
		self.feat_idx = read_obj_from_file(feat_map_path)

		label_map_path = os.path.join(model_dir, 'label_idx.txt')	
		self.label_idx = read_obj_from_file(label_map_path)

		return


	

	

	





	

	



