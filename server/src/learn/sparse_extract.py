#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import numpy
import scipy

from constants import *

# global variables: will be reconstructed everytime 'extract' is called

# mapping
inst_idx = None
feat_idx = None
label_idx = None

# matrices
feature_mat = None
i_mask = None
f_mask = None
i_target = None
f_target = None

def initialize_variables():
	inst_idx = {}
	feat_idx = {}
	label_idx = {}
	feature_mat = None
	i_mask = None
	f_mask = None
	i_target = None
	f_target = None

def extract_features(dir):
	# consider txt, feature annotation
	# fill in feature_mat, inst_idx, feat_idx
	pass

def extract_inst_labels(dir):
	# fill in label_idx, i_mask, i_target
	pass

def extract_feat_labels(dir):
	# fill in label_idx, f_mask, f_target
	pass

def extract(input_dir, output_dir):
	
	initialize_variables()

	extract_features(input_dir)

	extract_inst_labels(input_dir)

	extract_feat_labels(input_dir)

	# we can write the variables to disk (for debugging)
	# or we can just store them as variables in this module

