# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:38:25 2021

@author: gianm
"""

#%% Import Libraries
import pandas as pd
import re
import os
import glob
import math
import pickle
import numpy as np
from sklearn import preprocessing
#untuk Levenshtein distance
#import stringdist as sd
from nltk import sent_tokenize, word_tokenize
import threading

#%% Cleaner method
def cleaner(text):
    #remove newline
    text = text.replace('\n', ' ')
    #remove multiple spaces
    text = re.sub(' +', ' ', text)
    #remove special characters and numbers
    #text = re.sub('[^A-Za-z0-9\-\. ]', '', text)
    
    return text.lower()

#%% open dataset
doc1 = cleaner(open('dataset/3200004.txt').read())
doc2 = cleaner(open('dataset/3201100.txt').read())
doc3 = cleaner(open('dataset/3201121.txt').read())

#%% tokenize
sentences_doc1 = sent_tokenize(doc1)
sentences_doc2 = sent_tokenize(doc2)
sentences_doc3 = sent_tokenize(doc3)

word_list = set()

for i in sentences_doc1:
    for j in i.split():
        word_list.add(re.sub('^[^A-Za-z0-9]|[^A-Za-z0-9]+$', '', j))
        
for i in sentences_doc2:
    for j in i.split():
        word_list.add(re.sub('^[^A-Za-z0-9]|[^A-Za-z0-9]+$', '', j))
        
for i in sentences_doc3:
    for j in i.split():
        word_list.add(re.sub('^[^A-Za-z0-9]|[^A-Za-z0-9]+$', '', j))
        
#%% find idf
def computeIDF(word_list):
    N = 3
    word_idf = dict()
    for word in word_list:
        word_idf[word] = 0
        
    for word in word_list:
        #Hitung nx
        nx = 0
        if word in doc1:
            nx = nx + 1
        if word in doc2:
            nx = nx + 1
        if word in doc3:
            nx = nx + 1
        try:
            word_idf[word] = math.log2((N + 1)/nx)
        except:
            print(word)
            
    return word_idf

word_idf = computeIDF(word_list)

#%% TF 
def tf_doc1():
    global tf_doc1
    tf_doc1 = list()
    for sent in sentences_doc1:
        tf_dict = dict()
        for word in word_list:
            if word in sent:
                tf_dict[word] = math.log(sent.count(word) + 1) * word_idf[word]
        tf_doc1.append(tf_dict)
    print('tf_doc1 is finished')

def tf_doc2():
    global tf_doc2
    tf_doc2 = list()
    for sent in sentences_doc2:
        tf_dict = dict()
        for word in word_list:
            if word in sent:
                tf_dict[word] = math.log(sent.count(word) + 1) * word_idf[word]
        tf_doc2.append(tf_dict)
    print('tf_doc2 is finished')

def tf_doc3():
    global tf_doc3
    tf_doc3 = list()
    for sent in sentences_doc3:
        tf_dict = dict()
        for word in word_list:
            if word in sent:
                tf_dict[word] = math.log(sent.count(word) + 1) * word_idf[word]
        tf_doc3.append(tf_dict)
    print('tf_doc3 is finished')

threading.Thread(target=tf_doc1).start()
threading.Thread(target=tf_doc2).start()
threading.Thread(target=tf_doc3).start()


#%% normalization

vector_length = list()
for i in tf_doc1:
    length = math.sqrt(sum(n * n for n in i.values()))
    vector_length.append(length)
    
for i in tf_doc2:
    length = math.sqrt(sum(n * n for n in i.values()))
    vector_length.append(length)
    
for i in tf_doc3:
    length = math.sqrt(sum(n * n for n in i.values()))
    vector_length.append(length)


    
#%% DataFrame
df = pd.DataFrame()
df.insert(0, "Doc", 'doc')
df.insert(1, "Sentence", 'sent')

#%% Isi dataframe
def df_sent1():
    global df
    for sent in sentences_doc1:
        df = df.append({'Doc':1, 'Sentence': sent}, ignore_index=True)

def df_sent2():
    global df
    for sent in sentences_doc2:
        df = df.append({'Doc':2, 'Sentence': sent}, ignore_index=True)

def df_sent3():
    global df
    for sent in sentences_doc3:
        df = df.append({'Doc':3, 'Sentence': sent}, ignore_index=True)

df_sent1()
df_sent2()
df_sent3()

#%% kolom fitur
from dfply import *

zeros = [0.0] * len(df.index)
for i in word_list:
    df[i] = zeros

def fill_df1():
    df1 = df >> filter_by(X.Doc == 1)
    i = 0;
    for sent in tf_doc1:
        for word in sent.keys():
            df['{}'.format(word)][df1.index[i]] = sent[word]
        i = i + 1
    print('fill_df1 is finished')


def fill_df2():
    df2 = df >> filter_by(X.Doc == 2)
    i = 0;
    for sent in tf_doc2:
        for word in sent.keys():
            df['{}'.format(word)][df2.index[i]] = sent[word]
        i = i + 1
    print('fill_df2 is finished')

        
def fill_df3():
    df3 = df >> filter_by(X.Doc == 3)
    i = 0;
    for sent in tf_doc3:
        for word in sent.keys():
            df['{}'.format(word)][df3.index[i]] = sent[word]
        i = i + 1
    print('fill_df3 is finished')
            
threading.Thread(target=fill_df1).start()
threading.Thread(target=fill_df2).start()
threading.Thread(target=fill_df3).start()

#%% Load dataframe

df = pickle.load(open('df_main2.pickle', 'rb'))


#%% K-Means Clustering

from sklearn.cluster import KMeans
df2 = df >> drop(X.Doc, X.Sentence)

array_df = np.array(df2.values)

kmeans_model = KMeans(n_clusters=800, random_state=0).fit(array_df)

cluster_object = kmeans_model.labels_

centroids_set1 = kmeans_model.cluster_centers_

from sklearn.metrics import silhouette_score

coef_score_set1 = silhouette_score(array_df, cluster_object)