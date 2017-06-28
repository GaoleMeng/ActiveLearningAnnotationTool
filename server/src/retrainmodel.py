from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path

def dumpy(collection, document):

	file_path = u'./data' + collection + document
	file_path += u'.txt'
	f = open(file_path)
	Messager.error("hdhhd")
	Messager.error(file_path)
	for i, line in enumerate(f):
		Messager.error(line)
	f.close()

	Messager.error("back-end model trained")
	return {}

