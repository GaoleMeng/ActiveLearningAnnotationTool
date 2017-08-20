from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path
import json

from datetime import datetime

def label(label_array, collection, document):
	file_path = u'./data' +collection + document
	file_path += u'.lbl'

	f = open(file_path, 'a')


	with open("label.json") as json_file:
		json_data = json.load(json_file)
		# Messager.info("Instance labeled at "+str(json_data[1]))
		for i in range(json_data["num"]):
			if (label_array[i] == 'T'):
				f.write(str(json_data[str(i)]) + ":1" + " ")
			else:
				f.write(str(json_data[str(i)]) + ":0" + " ")
		f.write("\t" + str(datetime.now()) + "\n");

	f.close();

	#f.write(label1+" "+label2+" "+label3 + " "+str(datetime.now())+"\n")
	# f.close()
	Messager.info("Instance labeled at "+file_path)
	return {}

