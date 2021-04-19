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

#%% TF-IDF
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

#%% cari panjang vektor
def vector_1():
    global vector_length_1 
    vector_length_1 = list()
    for i in tf_doc1:
        length = math.sqrt(sum(n * n for n in i.values()))
        vector_length_1.append(length)
    print('vector_1 is finished')

def vector_2():
    global vector_length_2
    vector_length_2 = list()
    for i in tf_doc2:
        length = math.sqrt(sum(n * n for n in i.values()))
        vector_length_2.append(length)
    print('vector_2 is finished')

def vector_3():
    global vector_length_3
    vector_length_3 = list()
    for i in tf_doc3:
        length = math.sqrt(sum(n * n for n in i.values()))
        vector_length_3.append(length)
    print('vector_3 is finished')


threading.Thread(target=vector_1).start()
threading.Thread(target=vector_2).start()
threading.Thread(target=vector_3).start()

#%% normalisasi
def norm_1():
    for i, sent in enumerate(tf_doc1):
        for word in sent.keys():
            tf_doc1[i][word] = tf_doc1[i][word] / vector_length_1[i]

def norm_2():
    for i, sent in enumerate(tf_doc2):
        for word in sent.keys():
            tf_doc2[i][word] = tf_doc2[i][word] / vector_length_2[i]

def norm_3():
    for i, sent in enumerate(tf_doc3):
        for word in sent.keys():
            tf_doc3[i][word] = tf_doc3[i][word] / vector_length_3[i]
            
threading.Thread(target=norm_1).start()
threading.Thread(target=norm_2).start()
threading.Thread(target=norm_3).start()

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

#%% Save dataframe

pickle.dump(df, open('df_main2.pickle', 'wb'))

#%% Load dataframe

df = pickle.load(open('df_main2.pickle', 'rb'))

#%% K-Means Clustering

from sklearn.cluster import KMeans
df2 = df >> drop(X.Doc, X.Sentence)

array_df = np.array(df2.values)

kmeans_model = KMeans(n_clusters=1000, random_state=0).fit(array_df)

cluster_object_km = kmeans_model.labels_

centroids_km = kmeans_model.cluster_centers_

from sklearn.metrics import silhouette_score

coef_score_km = silhouette_score(array_df, cluster_object_km)

#%% Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
df2 = df >> drop(X.Doc, X.Sentence)

array_df = np.array(df2.values)
agglo_model = AgglomerativeClustering(n_clusters=1000, affinity='euclidean', linkage='single').fit(array_df)
    
#Array untuk label setiap baris
cluster_object_agg = agglo_model.labels_

from sklearn.metrics import silhouette_score
#Menghitung nilai Koefisien Silhouette
coef_score_agg = silhouette_score(array_df, cluster_object_agg)

#%% K-Means

def cosine_dist(x, y):
    return np.dot(x, y)

df['cluster'] = cluster_object_km

df_cluster = df >> filter_by(X.cluster == 9) >> select(X.Doc, X.Sentence)

df_group_cluster = (df 
                    >> group_by(X.cluster) 
                    >> summarize(count_sent = X.Doc.count())
                    >> filter_by(X.count_sent > 10))
                    
#%% compute distance
intra_cluster_dist = dict()
for i in df_group_cluster['cluster']:
    cluster_df = (df 
                  >> filter_by(X.cluster == i)
                  >> drop(X.Doc, X.Sentence, X.cluster))
    dist = 0
    m = 0;
    while m < len(cluster_df.index) - 1:
        n = m + 1
        while n < len(cluster_df.index):
            dist = dist + cosine_dist(cluster_df.iloc[m], cluster_df.iloc[n])
            n = n + 1
        m = m + 1
    intra_cluster_dist[i] = dist / len(cluster_df.index)
    
 

    