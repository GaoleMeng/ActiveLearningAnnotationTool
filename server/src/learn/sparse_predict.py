#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

from constants import *

def predict_instances(dir):
	pass

def explain_instances(input_dir, output_dir):
	# output feature predictions: will need the input files
	pass

def compute_metrics(output_dir):
	# current true loss, guessed loss
	# uncertainty measures on all instances
	pass


def predict(input_dir, output_dir):
	from sparse_extract import feature_mat

	predict_instances(output_dir)

	explain_instances(input_dir, output_dir)
	
	compute_metrics(output_dir)