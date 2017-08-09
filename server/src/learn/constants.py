#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

RAWTEXT_FILE_SUFFIX = 'txt'
PREDICT_FILE_SUFFIX = 'lbl'
EXPLAIN_FILE_SUFFIX = 'ann'
EXPLAIN_PREFIX = 'S'

UTILITY_POSITIVE_COEFF = 1.0
UTILITY_NEGATIVE_COEFF = -1.0
RANKED_LIST_THRESHOLD = 0.0

MAX_TOKEN_SEQUENCE_LENGTH = 500

STOP_AFTER_VAL_PERF_DROP_COUNT = 10

MAJORITY_PROB_MASS = 0.9

def make_smooth_label_dist(labels, true_label):
	if len(labels) < 2:
		ValueError('constants::make_smooth_label_dist(): number of labels should be at least 2.')
	prob = {}
	for l in labels:
		if l == true_label:
			prob[l] = MAJORITY_PROB_MASS
		else:
			prob[l] = (1 - MAJORITY_PROB_MASS) / (len(labels) - 1)
	return prob

EXP_STOPWORDS = dict.fromkeys([
"a", "an", "and", "are", "as", "at", "be", "but", "by",
"for", "if", "in", "into", "is", "it",
"no", "not", "of", "on", "or", "such",
"that", "the", "their", "then", "there", "these",
"they", "this", "to", "was", "will", "with"])