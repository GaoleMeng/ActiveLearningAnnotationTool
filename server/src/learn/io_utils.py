#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import sys
import os
import re
import codecs
import json

from constants import *
# from message import Messager

# ====== reader functions =======

def read_documents(input_dir):
	sys.stderr.write('io_utils::read_documents(): loading documents as token sequences ...\n')
	doc = {}
	for fn in os.listdir(input_dir):
		if fn.endswith('.{}'.format(RAWTEXT_FILE_SUFFIX)): # data file
			inst_name = fn[:-4]

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			fulltext = f.read()
			f.close()
			ss = re.sub(r'\s+', ' ', fulltext.strip()).split()

			doc[inst_name] = ss
	return doc

def read_instance_labels(input_dir):
	inst_labels = {}
	for fn in os.listdir(input_dir):
		if fn.endswith('.{}'.format(PREDICT_FILE_SUFFIX)): # label file
			inst_name = fn[:-4]

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			for line in f:
				pass # get the last line
			f.close()

			d = {}
			for ss in line.strip().split():
				sys.stderr.write(str(ss));
				k, v = ss.split(':')
				d[k] = float(v)
			inst_labels[inst_name] = d
	return inst_labels

def read_all_known_labels(input_dir):
	labels = {}
	for fn in os.listdir(input_dir):
		if fn.endswith('.{}'.format(PREDICT_FILE_SUFFIX)): # label file
			inst_name = fn[:-4]

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			for line in f:
				for ss in line.strip().split():
					l, p = ss.split(':')
					if l not in labels:
						labels[l] = 1
			f.close()
	for fn in os.listdir(input_dir):
		if fn.endswith('.{}'.format(EXPLAIN_FILE_SUFFIX)): # annotation file
			inst_name = fn[:-4]

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			for line in f:
				if (line[0] != 'S'):
					ann_id, span_info, feat_str = line.strip().split('\t')
					l, start, end = span_info.split()
					if l not in labels:
						labels[l] = 1
			f.close()
	return labels

def read_feature_labels(input_dir):
	all_known_labels = read_all_known_labels(input_dir)
	feat_labels = {}
	for fn in os.listdir(input_dir):
		if fn.endswith('.{}'.format(EXPLAIN_FILE_SUFFIX)): # annotation file
			inst_name = fn[:-4];

			fpath = os.path.join(input_dir, fn)
			f = open(fpath)
			for line in f:
				if (line[0] != 'S'):
					ann_id, span_info, feat_str = line.strip().split('\t')
					label, start, end = span_info.split()
					feat_labels[feat_str] = make_smooth_label_dist(all_known_labels, label)
			f.close()

	return feat_labels

def read_feature_subtasks(input_dir):
	return None

def read_pretrained_emb(input_dir):
	# now it's dummy
	return None


# ====== writer functions =======

def write_prediction(prediction, output_dir):
	for inst, l_dict in prediction.items():
		f = open( os.path.join(output_dir, '{}.{}'.format(inst, PREDICT_FILE_SUFFIX)), 'w' )
		pred_str = ' '.join([ '{}:{:.6f}'.format(k, v) for k, v in sorted(l_dict.items(), key = lambda x: x[0]) ])
		f.write( pred_str )
		f.close()
	return

def write_explanation(explanation, output_dir):
	for inst_name, inst_exp in explanation.items():
		f = open( os.path.join(output_dir, '{}.{}'.format(inst_name, EXPLAIN_FILE_SUFFIX )), 'w' )
		idx = 1
		for span, label_dict in inst_exp.items():
			start, end, feat = span
			for lbl, val in label_dict.items():
				span_info = '{}:{:.3f} {} {}'.format(lbl, val, start, end)
				f.write('{}{}\t{}\t{}\n'.format( EXPLAIN_PREFIX, idx, span_info, feat ))
				idx += 1
		f.close()
	return

def write_metrics(metrics, output_path):
	f = open(output_path, 'w')
	json.dump(metrics, f, sort_keys=True, indent=2)
	f.close()
	return

def write_inst_queries(inst_queries, output_dir):
	return