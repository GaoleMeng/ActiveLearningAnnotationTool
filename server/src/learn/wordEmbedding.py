import numpy as np
import os
from random import shuffle
import re
import tensorflow as tf;
import numpy as np;

from gensim.models import Word2Vec


def get_embeddding_matrix(documents):

	sentences = [];
	print "training word embedding...";
	for k, v in documents.items():
		sentences.append(v);
	model_doc = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
	return model_doc;


