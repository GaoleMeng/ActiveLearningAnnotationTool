#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

from os.path import isdir

from io_utils import *
from learner import InteractiveLearner
from evaluator import compute_metrics, select_inst_queries

# the main function
# what's possibly missing: a hold-out/validation set
def run(input_dir, output_dir, is_sparse = True, pretrained_emb = None):
	if not isdir(input_dir):
		exit('[Learn module] input directory does not exist: "{}"'.format(input_dir))
	if not isdir(output_dir):
		exit('[Learn module] output directory does not exist: "{}"'.format(output_dir))

	# read input data: documents, labels, and word embeddings
	documents = read_documents(input_dir)
	inst_labels = read_instance_labels(input_dir)
	feat_labels = read_feature_labels(input_dir)
	feat_tasks = read_feature_subtasks(input_dir)
	pretrained_emb = read_pretrained_emb(input_dir)

	# train model, predict labels, explain prediction
	learner = InteractiveLearner(is_sparse = True, pretrained_emb = None, encoder = 'one_hot', predictor = 'one_layer')
	learner.fit(documents, inst_labels, feat_labels, feat_tasks)
	prediction = learner.predict(documents)
	explanation = learner.explain(documents)
	
	# compute metrics, rank instances in active learning
	metrics = compute_metrics(inst_labels, prediction)
	inst_queries = select_inst_queries(inst_labels, prediction)

	# write out results
	write_prediction(prediction, output_dir)
	write_explanation(explanation, output_dir)
	write_model(learner, output_dir)
	write_metrics(metrics, output_dir)
	write_inst_queries(inst_queries, output_dir)


if __name__ == '__main__':
	import sys
	if len(sys.argv) != 3:
		exit('Params: input_dir output_dir')

	run(sys.argv[1], sys.argv[2])