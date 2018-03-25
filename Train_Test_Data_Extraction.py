#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Title : Aspect Based Sentiment Analysis (Par 1 - Test data generation)
# Authors : Kavya Danivas, Lilia Fkaier
# Dataset : Annotated Trip Advisor dataset http://nemis.isti.cnr.it/~marcheggiani/datasets/

import time
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import state_union
from matplotlib.cbook import unique
from nltk import pos_tag
from nltk.corpus import wordnet
# import gensim.models
# from gensim.models import Word2Vec
import os
import json

ps = PorterStemmer()                #to normalize into the root word
lemmatizer=WordNetLemmatizer()      #to give similar meaning of a word
tokenizer_reg = RegexpTokenizer(r'\w+') #to tokenize + removing punctuations
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# User input paramters
sim_val_threshold = 0.4 # Similarity threshold
path = 'LargeTripadvisor/'
senti_threshold = 3 # between 1 to 5, senti_threshold< is pos, else neg
writeFileName = './trainData.json'

# Predefine aspect terms in the file and associated aspect terms considered in the project
predefined_aspects = {"Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value", "Service":"service",
                      "Location": "location", "Sleep Quality":"sleep",
                      "Business service (e.g., internet access)":"business","Check in / front desk":"reception"}  #Business service (e.g., internet access)', 'Check in / front desk'

noun = ["NN", "NNS", "NNP", "NNPS"]             #defining POS tags for words #Currently not been used in the code
adjective = ["JJ", "JJR", "JJS"]                #defining POS tags for words

stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren'])
stop_words_modified=list(stop_words-a)
start_time = time.time()


# store aspect terms that are predefined
predefined_aspect_terms = predefined_aspects.keys()
predef_aspect_term_modified=unique(predefined_aspects.values())

# print aspect terms and categories that are predefined
print("Predefined Aspect Terms: ")
print(predefined_aspect_terms) #TODO: Clean the output

print("Defined Aspect Terms in the Project: ")
print(predef_aspect_term_modified) #TODO: Clean the output

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []


'''
Function to write to a json file
Input : Data - Text and label
Output : void
'''
def writeToJsonFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)


'''
Function to find for aspect terms and corresponding adjective
Input : Pos tagged words
Output : void
Comment : Called by process_content()
'''
def find_aspect_terms(wrds={}):
    max_sim_val = 0
    asp_word_found = False
    adj_found = False
    for (w1, t1) in wrds: # loop through all the pos tagged words
        for wn in aspect_terms_found:
            if(wn=='Overall'):
                continue

            wnn=predefined_aspects[wn]
            sim_val = 0
            wn2 = lemmatizer.lemmatize(w1)
            try:
                wone = wordnet.synset(wnn + '.n.01')
                wtwo = wordnet.synset(wn2 + '.n.01')
                sim_val = wone.wup_similarity(wtwo)
            except Exception:
                continue

            if(sim_val > sim_val_threshold and sim_val > max_sim_val):
                 asp_word = wn
                 max_sim_val = sim_val
                 asp_word_found = True

        if(t1 in adjective):
            adj_found=True

    if(asp_word_found): #checks if word token is in pre defined aspect term #TODO : More sophisticated condition if required (like finding similar nouns).
        return asp_word
    elif(adj_found):
        return 'Overall'
    else:
        asp_word = 'NOWORD'
        return asp_word


'''
Function for word tokenize and to provide pos tags
Input : sentence
Output : void
Comment : Called by extract_aspect_term(), Calls find_aspect_terms()
'''
def process_content(tokenized):
    try:
        for w in tokenized: # loop through all the tokenized sentence
            words = tokenizer_reg.tokenize(w)
            tagged = nltk.pos_tag(words) # tag pos to words
            aspect_term = find_aspect_terms(tagged) # Call find_aspect_terms
            return aspect_term
    except Exception as e:
        print(str(e))


'''
Function for cleaning the sentence and for sentence tokenize
Input : review line (paragraph or sentence)
Output : void
Comment : Calls process_content()
'''
def extract_aspect_term(review_line):
    review_line = review_line.lower() # normalize the sentence to all lower case
    # --------begin cleaning the sentence -------
    review_linewords = review_line.split()
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified] # remove all stop words from the sentence
    clean_review_line = ' '.join(filtered_review_line)
    # --------end cleaning the sentence ------- # not used
    sent_tokenizer=sent_tokenize(clean_review_line)
    aspect_term = process_content(sent_tokenizer) # Call process_content() to process the token to eventually find aspect terms
    return aspect_term


# There are 8113 files in Large Tripadvisor dataset
# It takes 7 hours to run the program with all the files
# k is the limited number of files
data_list=[]
k=0
# Collecting reviews for training
for infile in listing:
    curr_file = path+infile
    k+=1
    if k < 2:
        with open(curr_file) as data_file:
            data = json.load(data_file)
            reviews = data['Reviews']
            for rev in reviews:
                content = rev['Content']
                rating = rev['Ratings']
                aspect_terms_found = [ww for ww in rating]
                sentences = tokenizer.tokenize(content)
                for s in sentences:
                    asp_word_found = extract_aspect_term(s)
                    data={}
                    if(asp_word_found!='NOWORD' and asp_word_found!='Overall' and asp_word_found in aspect_terms_found):
                        label=rating[asp_word_found]
                        data['text']=s
                        data['label'] = 'pos' if int(label) > senti_threshold else 'neg'
                        data_list.append(data)
                    elif (asp_word_found == 'Overall' and 'Overall' in aspect_terms_found):
                        label = rating[asp_word_found]
                        data['text'] = s
                        data['label'] = 'pos' if float(label) > senti_threshold else 'neg'
                        data_list.append(data)


writeToJsonFile(writeFileName,data_list)