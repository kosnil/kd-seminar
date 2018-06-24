#!/usr/bin/env python
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import scipy
import json
import gensim
from pprint import pprint
from os import listdir
from os.path import isfile, join
from random import shuffle

retrain = False
dirPath = "../er_body_data/data/"
inputfile_path = "../er_body_data/data/article_bodies_2018-02-18_2018-02-28.json"


def main():
    docPaths = [f for f in listdir(dirPath) if f.endswith('.json')]
    print("Loading articles from:",docPaths)


    documents = []
    docLabels = []

    for doc in docPaths:
        with open(dirPath+doc, 'r') as fp:
            inputdata = json.load(fp)
        for company in inputdata.keys():
            for day in inputdata.get(company).keys():
                id = 0
                for article in inputdata.get(company).get(day):
                    docLabels.append(company+"_"+day+"_id"+str(id))
                    documents.append(normalize_text(article))
                    id += 1

    dictionary = dict(zip(docLabels, documents))

    print("Total no. of articles: %d" %(len(docLabels)))


    if retrain:
        print("---BUILD MODEL---")
        iterator = LabeledLineSentence(documents, docLabels)
        model = gensim.models.Doc2Vec(size=100, window=10, min_count=5, workers=11, alpha=0.025,  min_alpha=0.025)  # use fixed learning rate
        model.build_vocab(iterator)

        print("---TRAIN MODEL---")
        for epoch in range(10):
           #shuffle(iterator)
            model.train(iterator,total_examples=model.corpus_count, epochs=1)
            model.alpha -= 0.002  # decrease the learning rate
            #model.min_alpha = model.alpha  # fix the learning rate, no decay
            model.train(iterator, total_examples=model.corpus_count, epochs=1)
            print('Completed pass %i at alpha %f' % (epoch + 1, model.alpha))

        print("---SAVE MODEL---")
        model.save("doc2vec.model")

    else:
        print("---LOAD MODEL---")
        model = gensim.models.Doc2Vec.load("doc2vec.model")

    print("---TEST MODEL (Article)---") #comparison of some article (referecened by docLabels[20]) to its most similar, median-similar and least-similar article based on cosine similiarity
    similarities = model.docvecs.most_similar(docLabels[20],topn=model.docvecs.count)
    #print(model.docvecs[docLabels[20]])  #a sample vector

    print('\nTARGET article (%s):' % (docLabels[20]))
    print(dictionary.get(docLabels[20]))
    print("\nSimilar/dissimilar articles compared to the target article")
    for label, index in [('MOST similar', 0), ('MEDIAN similar', len(similarities) // 2), ('LEAST similar', len(similarities) - 1)]:
        print(u'\n%s %s' % (label, similarities[index]))
        print(dictionary.get(similarities[index][0]))

    print("\n---TEST MODEL (Word)---") #comparison of some word (here "word") to its 20 most similar words based on cosine similiarity
    word = "awesome"
    for w in model.most_similar(word, topn=20):
        print(w)


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])


def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    norm_text = norm_text.replace("\n", ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

main()



