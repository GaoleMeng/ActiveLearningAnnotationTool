from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path

from os import listdir

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

	for filename in listdir(outputpath):
		lines = "";
		if filename.endswith(".ann"):
			with open(file_path + filename, "r") as original_f:
				for line in original_f:
					vec = line.split("\t")
					if (not("EXP" in vec[0]) and not("S" in vec[0])):
						lines += line;
		if filename.endswith(".ann"):
			with open(outputpath + "/" + filename, "r") as f:
				with open(file_path + filename, "w") as f1:
					f1.write(lines);
					for line in f:
						f1.write(line)

	# original_f = open(file_path + document + ".ann", 'a');
	# with open(outputann, "r") as f:
	# 	for line in f:
	# 		original_f.write(line)

	# original_f.close();
	Messager.info("back-end model trained")
	return {}

	
