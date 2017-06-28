from __future__ import with_statement

from os.path import join as path_join
from message import Messager

import os.path

def label(label1,label2,label3, collection, document):

    file_path = u'./data' + collection + document
    file_path += u'.txt'
    f = open("labelfile.txt", 'w')
    f.write(document + " "+label1+" "+label2+" "+label3)
    # Messager.error(file_path)
    # for i, line in enumerate(f):
    #     Messager.error(line)
    # f.close()


    Messager.error(label1)
    Messager.error(label2)
    Messager.error(label3)




    return {}

    
