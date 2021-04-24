# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:20:27 2021

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
from nltk import sent_tokenize
import threading
from dfply import *

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
dataset = dict()
'''
dataset['doc1'] = cleaner(open('dataset/3200004.txt').read())
dataset['doc2'] = cleaner(open('dataset/3201100.txt').read())
dataset['doc3'] = cleaner(open('dataset/3201121.txt').read())
dataset['doc4'] = cleaner(open('dataset/3202016.txt').read())
dataset['doc5'] = cleaner(open('dataset/3202133.txt').read())
'''
data_name = ['doc{}'.format(i + 1) for i in range(10)]

for i, filename in enumerate(glob.glob(os.path.join('E:/Kuliah/Penambangan Data/plagiarism_detection/dataset', '*.txt'))):
   with open(os.path.join(os.getcwd(), filename), 'r', encoding='mbcs') as f: # open in readonly mode
      dataset[data_name[i]] = (cleaner(f.read()))

#%% tokenize
sentences = dict()
for doc in dataset.keys():
    sentences[doc] = sent_tokenize(dataset[doc])

word_list = set()
for doc in sentences.keys():
    for sent in sentences[doc]:
        for i in sent.split():
            word_list.add(re.sub('^[^A-Za-z0-9]|[^A-Za-z0-9]+$', '', i))

#%% hitung idf
def compute_IDF(word_list):
    N = len(dataset.keys())
    word_idf = dict()
    for word in word_list:
        word_idf[word] = 0
    
    for word in word_list:
        #Hitung nx
        nx = 0
        for doc in dataset.keys():
            if word in dataset[doc]:
                nx = nx + 1
        try:
            word_idf[word] = math.log2((N + 1) / nx)
        except:
            print(word)
            
    return word_idf

word_idf = compute_IDF(word_list)

#%% TF-IDF
tf_doc = dict()
def compute_TFIDF(doc):
    tf_doc[doc] = list()
    for sent in sentences[doc]:
        tf_dict = dict()
        for word in word_list:
            if word in sent:
                tf_dict[word] = math.log(sent.count(word) + 1) * word_idf[word]
        tf_doc[doc].append(tf_dict)
    print('tf_{} is finished '.format(doc))
    
for doc in dataset.keys():
    threading.Thread(target=compute_TFIDF, args=(doc,)).start()

#%% cari panjang vektor
vector_length = dict()
def compute_vector(doc):
    vector_length[doc] = list()
    for i in tf_doc[doc]:
        length = math.sqrt(sum(n * n for n in i.values()))
        vector_length[doc].append(length)
    print('vector_{} is finished '.format(doc))

for doc in dataset.keys():
    threading.Thread(target=compute_vector, args=(doc,)).start()

#%% normalisasi
def normalize(doc):
    for i, sent in enumerate(tf_doc[doc]):
        for word in sent.keys():
            tf_doc[doc][i][word] = tf_doc[doc][i][word] / vector_length[doc][i]
    print('normalize_{} is finished '.format(doc))

for doc in dataset.keys():
    threading.Thread(target=normalize, args=(doc,)).start()

#%% buat dataframe
df = pd.DataFrame()
df.insert(0, "Doc", 'doc')
df.insert(1, "Sentence", 'sent')

#%% isi dataframe
def fill_df(doc):
    global df
    for sent in sentences[doc]:
        df = df.append({'Doc':doc, 'Sentence': sent}, ignore_index=True)

for doc in dataset.keys():
    fill_df(doc)

#%% buat kolom fitur
zeros = [0.0] * len(df.index)
for i in word_list:
    df[i] = zeros

#%% isi kolom fitur
def fill_feature(doc):
    df_doc = (df
              >> filter_by(X.Doc == doc))
    
    for i, sent in enumerate(tf_doc[doc]):
        for word in sent.keys():
            df['{}'.format(word)][df_doc.index[i]] = sent[word]
    print('fill_feature_{} is finished'.format(doc))

for doc in dataset.keys():
    threading.Thread(target=fill_feature, args=(doc,)).start()

#%% cosine distance
def cosine_dist(x, y):
    return np.dot(x, y)

#%% Save dataframe
pickle.dump(df, open('../df_main3.pickle', 'wb'))

#%% Load dataframe
df = pickle.load(open('../df_main3.pickle', 'rb'))

#%% cari threshold
#dengan cara mengambil sample dari data frame dan mencari rata-rata jaraknya
from itertools import combinations

def find_threshold():
    sample_df = (df.sample(10)
                 >> drop(X.Doc, X.Sentence))
    dist = 0
    m = 0
    while m < len(sample_df.index) - 1:
        n = m + 1
        while n < len(sample_df.index):
            dist = dist + cosine_dist(sample_df.iloc[m], sample_df.iloc[n])
            n = n + 1
        m = m + 1
    return dist / len(list(combinations(sample_df.index, 2)))

#%% Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
df2 = df >> drop(X.Doc, X.Sentence)

array_df = np.array(df2.values)
agglo_model = AgglomerativeClustering(distance_threshold=find_threshold(), n_clusters=None).fit(array_df)
    
#Array untuk label setiap baris
cluster_object_agg = agglo_model.labels_

df['Cluster'] = cluster_object_agg
#%% grup tiap cluster
df_group_cluster = (df 
                    >> group_by(X.Cluster) 
                    >> summarize(count_sent = X.Doc.count())
                    >> filter_by(X.count_sent > 1))

#%% compute distance
intra_cluster_dist = dict()
for i in df_group_cluster['Cluster']:
    cluster_df = (df 
                  >> filter_by(X.Cluster == i)
                  >> drop(X.Doc, X.Sentence, X.Cluster))
    dist = 0
    m = 0;
    while m < len(cluster_df.index) - 1:
        n = m + 1
        while n < len(cluster_df.index):
            dist = dist + cosine_dist(cluster_df.iloc[m], cluster_df.iloc[n])
            n = n + 1
        m = m + 1
    intra_cluster_dist[i] = dist / len(list(combinations(cluster_df.index, 2)))

#%% semua cluster
cluster_all = dict()
for i in intra_cluster_dist.keys():
    cluster_all[i] = df >> filter_by(X.Cluster.isin([i])) >> select(X.Doc, X.Sentence)

#%% cluster (>1)
#cluster berisi anggota yang berasal dari dua atau lebih dokumen
two_or_more_docs = list()
for i in df_cluster_all.keys():
    df_doc = list(df_cluster_all[i].Doc)
    if len(set(df_doc)) > 1:
        two_or_more_docs.append(i)

cluster_candidates = dict()
for i in df_cluster_all.keys():
    if i in two_or_more_docs:
        cluster_candidates[i] = df_cluster_all[i]

#%% nilai kedekatan tiap dokumen
#dengan menggabungkan semua dokumen dari cluster-cluster yang terdapat di cluster_candidates
df_cluster = df >> filter_by(X.Cluster.isin(list(cluster_candidates.keys()))) >> drop(X.Cluster)

pair = dict()
docs = list(dataset.keys())
m = 0
while m < len(docs) - 1:  
    df_cluster_1 = (df_cluster
                   >> filter_by(X.Doc == docs[m])
                   >> drop(X.Doc, X.Sentence))
    n = m + 1
    while n < len(docs):
        df_cluster_2 = (df_cluster
                       >> filter_by(X.Doc == docs[n])
                       >> drop(X.Doc, X.Sentence))
        total = 0
        i = 0
        while i < len(df_cluster_1.index):
            j = 0
            while j < len(df_cluster_2.index):
                total += cosine_dist(df_cluster_1.iloc[i], df_cluster_2.iloc[j])
                j += 1
            i += 1
        pair['{f} - {s}'.format(f=docs[m], s=docs[n])] = total
        n += 1
    m += 1

#%% cari outlier
std = np.std(list(pair.values()))
mean = np.mean(list(pair.values()))

for i in pair.keys():
    if pair[i] > (mean + (3 * std)):
        print(i)
#%% pairing
#jika terdapat cluster dengan banyak anggota yang berasal dari dokumen yang berbeda-beda
#maka akan dihitung nilai kedekatan antar dokumen
df_one_cluster = df >> filter_by(X.Cluster.isin([0])) >> drop(X.Cluster)

pair = dict()
docs = list(dataset.keys())
m = 0
while m < len(docs) - 1:  
    df_cluster_1 = (df_one_cluster
                   >> filter_by(X.Doc == docs[m])
                   >> drop(X.Doc, X.Sentence))
    n = m + 1
    while n < len(docs):
        df_cluster_2 = (df_one_cluster
                       >> filter_by(X.Doc == docs[n])
                       >> drop(X.Doc, X.Sentence))
        total = 0
        i = 0
        while i < len(df_cluster_1.index):
            j = 0
            while j < len(df_cluster_2.index):
                total += cosine_dist(df_cluster_1.iloc[i], df_cluster_2.iloc[j])
                j += 1
            i += 1
        pair['{f} - {s}'.format(f=docs[m], s=docs[n])] = total
        n += 1
    m += 1
