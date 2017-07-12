#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

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
	def __init__(self, is_sparse = True, pretrained_emb = None, encoder = 'one_hot', predictor = 'one_layer', 
				 encoder_output_size = 100, softmax_input_size = 100):
		self.is_sparse = is_sparse
		self.pretrained_emb = pretrained_emb
		self.encoder_name = encoder
		self.predictor_name = predictor
		self.encoder_output_size = encoder_output_size
		self.softmax_input_size = softmax_input_size

		# internal params
		self._encoder = None
		self._predictor = None

		# mapping
		self._inst_idx = None    # size n dict
		self._feat_idx = None    # size m dict if is_sparse = True 
		self._label_idx = None   # size k dict

		# matrices
		self._feature_mat = None # n x m dict if is_sparse = True. n x d dict 
		self._i_mask = None
		self._f_mask = None
		self._i_target = None
		self._f_target = None
		self._g_target = None


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

		# extract_info

		# fit models

		return self

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
		pred = documents

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
		exp = documents
		return exp



	# The following two functions may not be straightforward if each time the model can have different features.
	def load_model(self, model_dir):
		""" Load the model from a folder.
		A model includes all attributes of the class


		"""

		return

	def save_model(self, model_dir):
		""" Save the model to a folder.
		A model includes all attributes of the class


		"""

		return
