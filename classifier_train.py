#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Title : Aspect Based Sentiment Analysis (Part 2 : Classifier)
# Authors : Kavya Danivas, Lilia Fkaier
# Dataset : Annotated Trip Advisor dataset http://nemis.isti.cnr.it/~marcheggiani/datasets/

import time
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
import pickle
import json
import random
from nltk.classify import NaiveBayesClassifier

# User input paramters
traintestdevidefactor = 0.7
writeFileName = './trainData.json'
writeWordSetFileName = './wordSetData.json'

naiivepickleFileName = './naive_trained_pickle'

label_cat = ['pos','neg']
stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren'])
stop_words_modified=list(stop_words-a)
start_time = time.time()

'''
Function to write to a json file
Input : Data - Text and label
Output : void
'''
def writeToJsonFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)


dataset = {}

with open(writeFileName, 'r') as fp:
    data = json.load(fp)
    document_tuple=[]
    all_words = []
    for label in label_cat:
        tokenized_sentences = []
        for list_item in data:
            if (list_item['label']==label):a
                s = list_item['text']
                s=s.lower()
                tokenized_sentence_stop = nltk.word_tokenize(s)
                tokenized_sentence = [wd for wd in tokenized_sentence_stop if not wd in stop_words]
                all_words.extend(tokenized_sentence)
                document_tuple.append((tokenized_sentence, label))
    random.shuffle(document_tuple)
    all_words=nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:8000]
    dataset=word_features
    def find_features (document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w]=(w in words)
        return features

    featuresets = [(find_features(rev), label) for (rev, label) in document_tuple]
    traintestdevide=int(traintestdevidefactor*(len(featuresets)))
    print(len(featuresets))
    print(traintestdevide)
    training_set = featuresets[traintestdevide:]
    testing_set = featuresets[:traintestdevide]

    # Naive bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Naive Bayes Algorithm Accuracy ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)

    save_classifier = open(naiivepickleFileName,"wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


writeToJsonFile(writeWordSetFileName,dataset)
