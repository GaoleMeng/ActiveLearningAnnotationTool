#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import os

from io_utils import *
from learner import InteractiveLearner, InteractiveLearnerNaiveBayes, DenseArch
from evaluator import compute_metrics, select_inst_queries

import tensorflow as tf

# the main function
# what's possibly missing: a hold-out/validation set
def run(input_dir, output_dir, is_sparse = True):
	if not os.path.isdir(input_dir):
		exit('[Learn module] input directory does not exist: "{}"'.format(input_dir))
	if not os.path.isdir(output_dir):
		exit('[Learn module] output directory does not exist: "{}"'.format(output_dir))

    # initialize = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(initialize)
    

	# read input data: documents, labels, and word embeddings
	documents = read_documents(input_dir)
	inst_labels = read_instance_labels(input_dir)
	feat_labels = read_feature_labels(input_dir)
	# feat_tasks = feat_labels.keys()
	feat_tasks = [];
	pretrained_emb = read_pretrained_emb(input_dir)

	archi = DenseArch.TWO_LAYER_AVG_EMB;
	model_dir = os.path.join(output_dir, 'model')
	# train model, predict labels, explain prediction
	# learner = InteractiveLearner(is_sparse = True, pretrained_emb = pretrained_emb, encoder = 'one_hot', predictor = 'one_layer')
	learner = InteractiveLearnerNaiveBayes(max_vocab = 10000, feat_label_pseudo_count = 10.)
	
	# learner = InteractiveLearner(is_sparse = False, max_vocab = 10000, dense_architecture = archi, pretrained_emb = None, learning_rate=1e-2)
	learner.fit(documents, inst_labels, feat_labels, feat_tasks, model_dir, None)
	sys.stderr.write('fitting finished\n')

	prediction = learner.predict(documents)
	explanation = learner.explain(documents, documents)
	
	# compute metrics, rank instances in active learning
	# metrics = compute_metrics(inst_labels, prediction)
	# inst_queries = select_inst_queries(inst_labels, prediction)

	# write out results
	write_prediction(prediction, output_dir)
	write_explanation(explanation, output_dir)
	
	# eval_txt = os.path.join(output_dir, 'eval.txt')
	# write_metrics(metrics, eval_txt)



	# write_inst_queries(inst_queries, output_dir)


if __name__ == '__main__':
	import sys
	if len(sys.argv) != 3:
		exit('Params: input_dir output_dir')

	run(sys.argv[1], sys.argv[2])