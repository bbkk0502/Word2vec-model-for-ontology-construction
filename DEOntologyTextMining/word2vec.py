#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:09:14 2020
This python script is for text preprocessing and word2vec language model development
Input: jSON files stored in data folder, TXT files stored in pdf folder
Output: Trained skipg model stored in working directory
@author: weinie
"""

import os, json
import pandas as pd
import re
import gensim 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import pos_tag
from gensim.models import Word2Vec 
from gensim.models.callbacks import CallbackAny2Vec

#--------------PREPARATION---------------------
#print loss after each epoch of word2vec training
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

#STEP 1: READ DOWNLOADED FULL-TEXT FROM FOLDER
#THIS includes JSON files downloaded from Elsevier and text files from other publishers' websites
#-------Step 1.1 read json files that stored in 'data' folder to textList[]
textList=[]
path_to_json = 'data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
for file in json_files:
    json_file=open('data/'+file) #read json file stored in data folder
    data = json.load(json_file)
    textList.append(data['originalText'])
    
#-------Step 1.2 read text files that stored in 'pdf' folder to textList2[]
textList2=[]
path_to_txt = 'pdf'
txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
for file in txt_files:
    txt_file=open('pdf/'+file) #read json file stored in data folder
    textList2.append(txt_file.read())
#-------Step 1.3 combine two list to textListProcessed
textListProcessed=textList+textList2


#STEP2: TEXT PRE-PROCESSING
#-------Step 2.1 transform to lower case
for i in range(len(textListProcessed)):
    textListProcessed[i] = textListProcessed[i].lower()
#-------Step 2.2 replace keyword to standardise spellings; replace regex characters such as \n; remove - in a word
for i in range(len(textListProcessed)):
    textListProcessed[i] = textListProcessed[i].replace("\n"," ")
    textListProcessed[i] = textListProcessed[i].replace("- ","")
    textListProcessed[i] = textListProcessed[i].replace("test bed","testbed")
    textListProcessed[i] = textListProcessed[i].replace("testbeds","testbed")
    textListProcessed[i] = textListProcessed[i].replace("pilot plant","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot-plant","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot_plants","pilot-plant")
    textListProcessed[i] = textListProcessed[i].replace("living lab","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living laboratory","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living laboratories","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living-lab","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("living_labs","living-lab")
    textListProcessed[i] = textListProcessed[i].replace("innovation lab","innovation-lab")
    textListProcessed[i] = textListProcessed[i].replace("innovation hub","innovation-hub")
    textListProcessed[i] = textListProcessed[i].replace("innovation space","innovation-space")
    textListProcessed[i] = textListProcessed[i].replace("innovation center","innovation-center")
    textListProcessed[i] = textListProcessed[i].replace("innovation centre","innovation-center")
    textListProcessed[i] = textListProcessed[i].replace("pilot scale plant","pilot-scale-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot-scale plant","pilot-scale-plant")
    textListProcessed[i] = textListProcessed[i].replace("pilot line","pilot-line")
    textListProcessed[i] = textListProcessed[i].replace("pilot facility","pilot-facility")
    textListProcessed[i] = textListProcessed[i].replace("pilot facilities","pilot-facility")
    textListProcessed[i] = textListProcessed[i].replace("pilot lab","pilot-lab")
    textListProcessed[i] = textListProcessed[i].replace("pilot laboratory","pilot-lab")
    textListProcessed[i] = textListProcessed[i].replace("pilot demonstration","pilot-demonstration")
    textListProcessed[i] = textListProcessed[i].replace("prototype platform","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototype platforms","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototyping platform","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototyping platforms","prototyping-platform")
    textListProcessed[i] = textListProcessed[i].replace("prototype plant","prototype-plant")
    textListProcessed[i] = textListProcessed[i].replace("prototype plants","prototype-plant")
    textListProcessed[i] = textListProcessed[i].replace("demonstration facility","demonstration-facility")
    textListProcessed[i] = textListProcessed[i].replace("demonstration facilities","demonstration-facility")
    textListProcessed[i] = textListProcessed[i].replace("demonstration center","demonstration-center")
    textListProcessed[i] = textListProcessed[i].replace("demonstration centre","demonstration-center")
    textListProcessed[i] = textListProcessed[i].replace("open innovation","open-innovation")
    textListProcessed[i] = textListProcessed[i].replace("scaleup","scale-up")
    textListProcessed[i] = textListProcessed[i].replace("scale up","scale-up")
    textListProcessed[i] = textListProcessed[i].replace("field laboratory","field-lab")
    textListProcessed[i] = textListProcessed[i].replace("field lab","field-lab")
    textListProcessed[i] = textListProcessed[i].replace("test bench","test-bench")
    textListProcessed[i] = textListProcessed[i].replace("urban laboratory","urban-lab")
    textListProcessed[i] = textListProcessed[i].replace("urban lab","urban-lab")
    textListProcessed[i] = textListProcessed[i].replace("city laboratory","city-lab")
    textListProcessed[i] = textListProcessed[i].replace("city lab","city-lab")
    textListProcessed[i] = textListProcessed[i].replace("maker space","maker-space")
    textListProcessed[i] = textListProcessed[i].replace("test bench","test-bench")
    textListProcessed[i] = textListProcessed[i].replace("test benches","test-bench")    
    textListProcessed[i] = textListProcessed[i].replace("demonstration plant","demonstration-plant") 
    textListProcessed[i] = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', textListProcessed[i])


#-------Step 2.3 Tokenization using NLTK tokenizer
for i in range(len(textListProcessed)):
    textListProcessed[i]=word_tokenize(textListProcessed[i])

#-------Step 2.4 r\Remove stopwords, digits, and words with length < 4 and length > 25; need to hardcode to ensure pre-coded words are not filtered out by wordnet english word recogniser
stop_words=set(stopwords.words("english"))#build stopword table
ProcessList2=textListProcessed
for i in range(len(textListProcessed)):
    filtered_sent=[]
    for w in textListProcessed[i]:
        if w not in stop_words and len(w)>4 and len(w)<25 and (wordnet.synsets(w) or w =='testbed' or w=='test-bench' or w=='maker-space' or w=='urban-lab'or w=='city-lab'or w=='field-lab'or w=='living-lab' or w=='pilot-plant' or w=='test-bench' or w=='sandbox' or w=='pilot-line' or w=='innovation-lab' or w=='innovation-hub' or w=='innovation-space' or w=='innovation-center' or w=='pilot-scale-plant' or w=='pilot-line' or w=='pilot-facility' or w=='pilot-lab' or w=='pilot-demonstration' or w=='prototyping-platform' or w=='prototype-plant' or w=='demonstration-facility' or w=='demonstration-center' or w=='open-innovation' or w=='scale-up'):
            filtered_sent.append(w)
        ProcessList2[i]=filtered_sent

#-------Step 2.5 Lemmentization based on POS tag
#build dictionary
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
lemmatizer = WordNetLemmatizer() 
for i in range(len(ProcessList2)):
    lemmatize_words=[]
    for token, tag in pos_tag(ProcessList2[i]):
        lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
        lemmatize_words.append(lemma)
    ProcessList2[i]=lemmatize_words

#-------Step 2.6 Calculate total # of unique words included 
unique_words=[]
total_count=0
for doc in ProcessList2:
    total_count=total_count+len(doc)
    for word in doc:
        unique_words.append(word)
print(len(set(unique_words)))

#STEP 3: WORD2VEC SKIP-G MODEL TRAINING
skipg = Word2Vec(ProcessList2, sg=1, workers=10, min_count=30, window=6, compute_loss=True, seed=1, callbacks=[callback()])
words = list(skipg.wv.vocab) #get vectorised vocabulary list
skip.wv.most_similar('testbed',topn=100) #use this function to find top 100 similar word of 'testbed' (can be changed)
skipg.save("skipg.model") #save trained skipg model in current working directory


#Additional Step: Co-occurrence matrix

import numpy as np
import nltk
from nltk import bigrams
import itertools
import pandas as pd
 
 
def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))
 
    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
 
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
 
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)
 
    # return the matrix and the index
    return co_occurrence_matrix, vocab_index
 

text_data=ProcessList2
 
# Create one list using many lists
data = list(itertools.chain.from_iterable(text_data))
matrix, vocab_index = generate_co_occurrence_matrix(data)
 
 
data_matrix = pd.DataFrame(matrix, index=vocab_index,
                             columns=vocab_index)

#search all co-occuring word of 'test-bench'
i=0
for val in data_matrix["test-bench"]: 
    if val>0:#this value 0 can be adjusted
        print(data_matrix.columns.values[i],'-> ',val)
    i=i+1


