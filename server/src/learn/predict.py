#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import sys, os
from learner import InteractiveLearner, InteractiveLearnerNaiveBayes
from io_utils import *

if len(sys.argv) != 4:
	exit('Params: input_dir model_dir output_dir')

input_dir = sys.argv[1]
model_dir = sys.argv[2]
output_dir = sys.argv[3]

if not os.path.exists(input_dir):
	exit('Input directory does not exist: "{}"'.format(input_dir))

if not os.path.exists(model_dir):
	exit('Model directory does not exist: "{}"'.format(model_dir))

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

documents = read_documents(input_dir)

learner = InteractiveLearner()
# learner = InteractiveLearnerNaiveBayes()
learner.load_model(model_dir)

prediction = learner.predict(documents)
explanation = learner.explain(documents, documents)

write_prediction(prediction, output_dir)
write_explanation(explanation, output_dir)