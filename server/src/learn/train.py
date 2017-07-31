#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import sys, os
from learner import InteractiveLearner
from io_utils import *

if len(sys.argv) != 4:
	exit('Params: param_str input_dir model_dir')

param_str = sys.argv[1]
input_dir = sys.argv[2]
model_dir = sys.argv[3]

if not os.path.exists(input_dir):
	exit('Input directory does not exist: "{}"'.format(input_dir))

documents = read_documents(input_dir)
inst_labels = read_instance_labels(input_dir)
feat_labels = read_feature_labels(input_dir)
feat_tasks = feat_labels.keys()
# feat_tasks = []

is_sparse = 'sparse' in param_str
learner = InteractiveLearner(is_sparse = is_sparse, pretrained_emb = None, encoder = 'one_hot', predictor = 'one_layer')
learner.fit(documents, inst_labels, feat_labels, feat_tasks)

learner.save_model(model_dir)