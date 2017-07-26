#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

import sys, os, json
from io_utils import *
from evaluator import *

if len(sys.argv) != 4:
	exit('Params: input_dir output_dir eval_txt')

input_dir = sys.argv[1]
output_dir = sys.argv[2]
eval_txt = sys.argv[3]

inst_labels = read_instance_labels(input_dir)
predictions = read_instance_labels(output_dir)
metrics = compute_metrics(inst_labels, predictions)
write_metrics(metrics, eval_txt)