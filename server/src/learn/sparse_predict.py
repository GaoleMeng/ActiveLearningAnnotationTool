#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import os

from constants import *
from message import Messager
from datetime import datetime

def predict_instances(feature_mat, inst_idx, label_idx, output_dir):
	
	# dummy implementation
	import random
	random.seed(datetime.now())
	for inst in inst_idx:
		f = open( os.path.join(output_dir, '{}.{}'.format(inst, PREDICT_FILE_SUFFIX)), 'w' )
		label_dist = [random.uniform(0, 1) for i in range(3)]
		s = sum(label_dist)
		label_dist = [i/s for i in label_dist]
		f.write('label1\t{}\nlabel2\t{}\nlabel3\t{}\n'.format(label_dist[0], label_dist[1], label_dist[2]))	
		f.close()

def explain_instances(input_dir, output_dir):
	# output feature predictions: will need the input files
	
	# dummy implementation: higlight the first 5 words
	from random import choice
	for fn in os.listdir(input_dir):
		if fn.endswith('.txt'): # data file
			inst_name = fn[:-4]
			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			fulltext = f.read()
			f.close()
			ss = fulltext.split()

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

def compute_metrics(output_dir):
	# current true loss, guessed loss
	# uncertainty measures on all instances
	pass


def predict(input_dir, output_dir):
	from sparse_extract import feature_mat, inst_idx, label_idx

	predict_instances(feature_mat, inst_idx, label_idx, output_dir)

	explain_instances(input_dir, output_dir)
	
	compute_metrics(output_dir)