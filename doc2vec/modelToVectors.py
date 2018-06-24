#!/usr/bin/env python
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import json
import gensim
from collections import defaultdict
import datetime
from dateutil import parser

print("---LOAD MODEL---")
model = gensim.models.Doc2Vec.load("doc2vec.model")
mindate = datetime.datetime.max
maxdate = datetime.datetime.min


result = defaultdict(dict)
for k in model.docvecs.doctags.keys():
    company,date,id = k.split('_')
    dt = parser.parse(date)
    if dt > maxdate:
        maxdate = dt
    if dt < mindate:
        mindate = dt

    if date in result[company].keys():
        result[company][date].append(model.docvecs[k].tolist())
    else:
         result[company][date] = model.docvecs[k].tolist()


PATH = "data/article_vectors_" + str(mindate.date()) + "-" + str(maxdate.date()) + ".json"

print("---CREATING FILE %s---" %(PATH))
with open(PATH, 'w') as fp:
     json.dump(result, fp)

