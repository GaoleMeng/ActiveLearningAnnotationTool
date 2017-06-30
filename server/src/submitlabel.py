from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path

from datetime import datetime

def label(label1,label2,label3, collection, document):
    file_path = u'./data' +collection + document
    file_path += u'.lbl'

    f = open(file_path, 'a')
    f.write(label1+" "+label2+" "+label3 + " "+str(datetime.now())+"\n")
    f.close()
    Messager.info("Instance labeled at "+file_path)
    return {}

