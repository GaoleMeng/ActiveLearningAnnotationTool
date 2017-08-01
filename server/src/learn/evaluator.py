#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

from copy import deepcopy
from constants import *

def get_avg_precs(label, pred):
	# true_by_label[l][i] = 1
	true_by_label = {}
	for l in label.items()[0][1].keys():
		true_by_label[l] = {}
	for i, l_dict in label.items():
		l = sorted( l_dict.items(), key = lambda x: x[1], reverse = True )[0][0]
		true_by_label[l][i] = 1
	# pred_by_label[l][i] = score
	pred_by_label = {}
	for l in pred.items()[0][1].keys():
		pred_by_label[l] = {}
	for i, p in pred.items():
		for l, s in p.items():
			if s >= RANKED_LIST_THRESHOLD: # set threshold to compute MAP
				pred_by_label[l][i] = s
	# calculate ap for each true label
	ap = {}
	util = {} # utility in information filtering
	for l in true_by_label:
		if l in pred_by_label and len(pred_by_label[l]) > 0: # the dictionary can be empty because of the thresholding
			sum_prec = 0.0
			num_recall_pt = 0.0
			curr_pos = 1
			sum_util = 0.0
			for e in sorted(pred_by_label[l].iteritems(), key = lambda x: x[1], reverse = True):
				if e[0] in true_by_label[l]:
					num_recall_pt += 1
					sum_prec += num_recall_pt / curr_pos
					sum_util += UTILITY_POSITIVE_COEFF
				else:
					sum_util += UTILITY_NEGATIVE_COEFF
				curr_pos += 1
			ap[l] = sum_prec / (len(true_by_label[l]) + 1e-50)

			util[l] = sum_util / (len(pred_by_label[l]) + 1e-50)
		else:
			ap[l] = 0.0
			util[l] = UTILITY_NEGATIVE_COEFF
	return ap, util

# inst_label[inst_id][label_id] = y(label|instance)
# prediction[inst_id][label_id] = p(label|instance)
metric_template = {'tp': 1e-100, 'tn': 1e-100, 'fp': 1e-50, 'fn': 1e-50}
def compute_metrics(label, pred):
	total = 0
	correct = 0
	d = {}

	# figure out all true labels in test data
	for i, l_dict in label.items():
		for l, p in l_dict.items():
			if l not in d:
				d[l] = deepcopy(metric_template)

	for id, l_dict in label.items():
		true_label = sorted( l_dict.items(), key = lambda x: x[1], reverse = True )[0][0]
		if id in pred:
			guessed_arr = sorted( pred[id].items(), key = lambda x: x[1], reverse = True )
			# print guessed_arr
			guessed_label = guessed_arr[0][0]
			if guessed_label == true_label:
				correct += 1
				for l in d:
					if l == true_label:
						d[l]['tp'] += 1
					else:
						d[l]['tn'] += 1
			else:
				if guessed_label in d:
					d[true_label]['fn'] += 1
					d[guessed_label]['fp'] += 1

			total += 1
		else:
			raise ValueError ('evaluator::compute_metrics(): data_id {} is not predicted.'.format(id))

	# compute micro-averaged precision, recall, f1
	all_tp = sum([d[l]['tp']  for l in d])
	all_fp = sum([d[l]['fp']  for l in d])
	all_fn = sum([d[l]['fn']  for l in d])
	
	# micro_avg_precision = all_tp / (all_tp + all_fp)
	# micro_avg_recall    = all_tp / (all_tp + all_fn)
	# micro_avg_f1 = 2*micro_avg_precision*micro_avg_recall / (micro_avg_precision + micro_avg_recall)
	# micro_avg_f05 = (1+0.5**2)*micro_avg_precision*micro_avg_recall / ((0.5**2)*micro_avg_precision + micro_avg_recall)

	# compute macro-averaged precision, recall, f1
	for l in d:
		d[l]['precision'] = d[l]['tp'] / (d[l]['tp'] + d[l]['fp'])
		d[l]['recall']    = d[l]['tp'] / (d[l]['tp'] + d[l]['fn'])
		d[l]['f1']        = 2*d[l]['precision']*d [l]['recall'] / (d[l]['precision'] + d[l]['recall'])

	macro_avg_precision = sum([ d[l]['precision'] for l in d ]) / len(d)
	macro_avg_recall = sum([ d[l]['recall'] for l in d ]) / len(d)
	macro_avg_f1 = sum([ d[l]['f1'] for l in d ]) / len(d)

	accuracy = float(correct) / float(total)

	avg_precs, avg_utils = get_avg_precs(label, pred)

	# print avg_precs
	# print avg_utils
	mean_ap = sum(avg_precs.values()) / len(avg_precs)
	# mean_util = sum(avg_utils.values()) / len(avg_utils)

	return {
	'macro_avg_precision': macro_avg_precision
	,'macro_avg_recall': macro_avg_recall
	,'macro_avg_f1': macro_avg_f1
	,'accuracy': accuracy
	,'mean_ap': mean_ap
	,'per_label_ap': avg_precs
	,'per_label_prf': d
	# ,'mean_util': mean_util
	# ,'micro_avg_precision': micro_avg_precision
	# ,'micro_avg_recall': micro_avg_recall
	# ,'micro_avg_f1': micro_avg_f1
	# ,'micro_avg_f05': micro_avg_f05
	}

def select_inst_queries(inst_labels, prediction):
	return
