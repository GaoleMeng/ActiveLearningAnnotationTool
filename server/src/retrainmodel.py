from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path
from learn.main import run

def dumpy(collection, document):

	file_path = u'./data' + collection
	outputpath = u'./data' + collection + "output"
	# f = open(file_path)
	# Messager.error("hdhhd")
	# Messager.error(file_path)
	# for i, line in enumerate(f):
	# 	Messager.error(line)
	# f.close()
	run(file_path, outputpath)
	outputlbl = outputpath + "/" + document + ".lbl";
	outputann = outputpath + "/" + document + ".ann";


	
	original_f = open(file_path + document + ".ann", 'a');
	with open(outputann, "r") as f:
		for line in f:
			original_f.write(line)

	original_f.close();
	Messager.info("back-end model trained")
	return {}

	
