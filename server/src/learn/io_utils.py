#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import os
import re

from constants import *
# from message import Messager

# ====== reader functions =======

def read_documents(input_dir):
	doc = {}
	for fn in os.listdir(input_dir):
		if fn.endswith('.txt'): # data file
			inst_name = fn[:-4]

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			fulltext = f.read()
			f.close()
			ss = re.sub(r'\s+', ' ', fulltext.strip()).split()

			doc[inst_name] = ss
	return doc

def read_instance_labels(input_dir):
	return None

def read_feature_labels(input_dir):
	return None

def read_feature_subtasks(input_dir):
	return None

def read_pretrained_emb(input_dir):
	# now it's dummy
	return None


# ====== writer functions =======

def write_prediction(prediction, output_dir):
	# dummy implementation
	import random
	from datetime import datetime
	random.seed(datetime.now())
	for inst in prediction:
		f = open( os.path.join(output_dir, '{}.{}'.format(inst, PREDICT_FILE_SUFFIX)), 'w' )
		label_dist = [random.uniform(0, 1) for i in range(3)]
		s = sum(label_dist)
		label_dist = [i/s for i in label_dist]
		f.write('label1\t{}\nlabel2\t{}\nlabel3\t{}\n'.format(label_dist[0], label_dist[1], label_dist[2]))	
		f.close()
	return

def write_explanation(explanation, output_dir):
	# dummy implementation: higlight the first 5 words
	from random import choice
	for inst_name, ss in explanation.items():

		f = open( os.path.join(output_dir, '{}.{}'.format(inst_name, EXPLAIN_FILE_SUFFIX )), 'w' )
		end = -1
		idx = 1
		for i in [0,1,2,3,4]:
			start = end + 1
			end = start + len(ss[i])
			l = choice([1,2,3])
			span_info = '{}:label{} {} {}'.format(EXPLAIN_PREFIX, l, start, end)
			f.write('{}{}\t{}\t{}\n'.format( EXPLAIN_PREFIX, idx, span_info, ss[i] ))

			idx += 1
		f.close()
	return

def write_model(learner, output_dir):
	return

def write_metrics(metrics, output_dir):
	return

def write_inst_queries(inst_queries, output_dir):
	return