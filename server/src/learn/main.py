#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:



# the main function
# what's possibly missing: a hold-out/validation set
def run(input_dir, output_dir, is_sparse = True, pretrained_emb = None):
	if is_sparse:
		from sparse_extract import extract
		from sparse_train import train
		from sparse_predict import predict

		extract(input_dir, output_dir)
		train(output_dir)
		predict(input_dir, output_dir)
	else:
		# dense version: should be symmetric to sparse, except it takes a pretrained embedding file
		pass


if __name__ == '__main__':
	import sys
	if len(sys.argv) != 3:
		exit('Params: input_dir output_dir')

	run(sys.argv[1], sys.argv[2])