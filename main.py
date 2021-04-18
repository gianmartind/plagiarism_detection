#%% Import Library

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

#%% Text Cleaner

def cleaner(text):
    #remove newline
    text = text.replace('\n', ' ')
    #remove multiple spaces
    text = re.sub(' +', ' ', text)
    #remove special characters and numbers
    text = re.sub('[^A-Za-z0-9\-\. ]', '', text)
    
    return text.lower()

 #%% Import Dataset

documents = list()
#Import semua file .txt di folder dataset
for filename in glob.glob(os.path.join('E:/Kuliah/Penambangan Data/plagiarism_detection/dataset', '*.txt')):
   with open(os.path.join(os.getcwd(), filename), 'r', encoding='mbcs') as f: # open in readonly mode
      documents.append(cleaner(f.read()))
      
#Kamus Indonesia
bahasa = open('E:/Kuliah/Penambangan Data/plagiarism_detection/models/Indonesia.txt', 'r')
bahasa = bahasa.read().split('\n')

#%% Pisah kalimat
sentence_list = list()
for doc in documents:
    sentences = doc.split('. ')
    sentence_ = list()
    for i in sentences:
        if len(i.split()) > 3:
            i = re.sub('^ ', '', i)
            sentence_.append(re.sub(' +', ' ', i))
    sentence_list.append(sentence_)

#%% Hitung semua kata yang muncul

def computeAllWords(docs):
    all_words = set()
    for text in docs:
        for word in word_tokenize(text):
            all_words.add(re.sub('\.$','', word)) 
    return all_words

allWords = computeAllWords(documents)

pickle.dump(allWords, open('allWords.pickle', 'wb'))

#%% Load all words

allWords = pickle.load(open('allWords.pickle', 'rb'))

#%% Hitung IDF tiap kata

def computeIDF(allWords):
    N = len(documents)
    words_idf = dict()
    for word in allWords:
        words_idf[word] = 0
        
   
    for word in allWords:
        #Hitung nx
        nx = 0
        for doc in documents:
            if word in doc:
                nx += 1
        try:
            words_idf[word] = math.log2((N + 1)/nx)
        except:
            print(word)
            
    return words_idf

allWords_idf = computeIDF(allWords)

pickle.dump(allWords_idf, open('allWords_idf.pickle', 'wb'))

#%% Load all words idf

allWords_idf = pickle.load(open('allWords_idf.pickle', 'rb'))

#%% Buat dataframe dengan semua fitur

df = pd.DataFrame()
zeros = [0.0] * 30
for i in allWords_idf.keys():
    df[i] = zeros

#Isi dataframe dengan TF-IDF
for i in range(len(df.index)):
    for word in allWords_idf.keys():
        df['{}'.format(word)][i] = math.log10(documents[i].count(word) + 1) * allWords_idf[word]
        
pickle.dump(df, open('data_frame.pickle', 'wb'))

#%% Load data frame

df = pickle.load(open('data_frame.pickle', 'rb'))

#%% Dataframe untuk vektor kalimat

df = pd.DataFrame()
df.insert(0, "Doc", "doc")
df.insert(1, "Sentence", 'sent')

#%% Isi dataframe

i = 0
for doc in sentence_list:
    for sent in doc:
        df = df.append({'Doc':i, 'Sentence': sent}, ignore_index=True)
    i += 1
 
zeros = [0.0] * len(df.index)
for i in allWords_idf.keys():
    df[i] = zeros
    
for i in range(len(df.index)):
    for word in allWords_idf.keys():
        df['{}'.format(word)][i] = math.log10(df[][i].count(word) + 1) * allWords_idf[word]
        
pickle.dump(df, open('data_frame2.pickle', 'wb'))
#%% Load data frame

df = pickle.load(open('data_frame2.pickle', 'rb'))

#%% Normalisasi Vektor

vector_length = list()
for row in df.index:
    v_len = 0
    for col in df.loc[row]:
        v_len += col * col
    vector_length.append(math.sqrt(v_len))

df = df.div(vector_length, axis=0)

#%% Normalisasi sklearn

for i in range(len(df.index)):
    df.iloc[i] = preprocessing.normalize(df.iloc[i][:,np.newaxis], axis=0).ravel()

pickle.dump(df, open('data_frame_normalized.pickle', 'wb'))

#%% Cosine distance

def cosine_dist(x, y):
    return np.dot(x, y)

#%% Hitung Cosine distance

dist_df = pd.DataFrame(0, index=np.arange(0, 30), columns=np.arange(0, 30))

for i in np.arange(0, len(df.index)):
    for j in np.arange(0, len(df.index)):
        dist_df.iloc[i, j] = cosine_dist(list(df.iloc[i]), list(df.iloc[j]))

#%% Segitiga bawah

lower_tri = np.tril(dist_df)

#%% K-Means Clustering

from sklearn.cluster import KMeans

array_df = np.array(df.values)

kmeans_model = KMeans(n_clusters=4, random_state=0).fit(array_df)

cluster_object = kmeans_model.labels_

centroids_set1 = kmeans_model.cluster_centers_

from sklearn.metrics import silhouette_score

coef_score_set1 = silhouette_score(array_df, cluster_object)

#%% K-Means (2)

from dfply import *

df['cluster'] = cluster_object

df_cluster_1 = df >> filter_by(X.cluster == 1) >> drop(X.cluster)
cluster_1_dists = list()

i = 0
while i < len(df_cluster_1.index) - 1:
    j = i + 1
    while j < len(df_cluster_1.index):
        cluster_1_dists.append(dist_df.iloc[df_cluster_1.index[i], df_cluster_1.index[j]])
        j = j + 1
    i = i + 1

cluster_1_dists_sum = sum(i for i in cluster_1_dists)


