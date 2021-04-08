#%%
# Import Library
import pandas as pd
import re
import os
import glob
import math
import pickle

#%%
#Text Cleaner
def cleaner(text):
    #remove newline
    text = text.replace('\n', ' ')
    #remove multiple spaces
    text = re.sub(' +', ' ', text)
    #remove special characters and numbers
    text = re.sub('[^A-Za-z\- ]', '', text)
    
    return text.lower()

 #%% 
#Import Dataset
documents = list()
#Import all .txt file in dataset folder
for filename in glob.glob(os.path.join('E:/Kuliah/Penambangan Data/plagiarism_detection/dataset', '*.txt')):
   with open(os.path.join(os.getcwd(), filename), 'r', encoding='mbcs') as f: # open in readonly mode
      documents.append(cleaner(f.read()))
      
#Kamus Indonesia
bahasa = open('E:/Kuliah/Penambangan Data/plagiarism_detection/models/Indonesia.txt', 'r')
bahasa = bahasa.read().split('\n')

#%%
#Hitung semua kata yang muncul
def computeAllWords(docs):
    all_words = set()
    for text in docs:
        for word in text.split():
            if word in bahasa:
                all_words.add(word)    
    return all_words

allWords = computeAllWords(documents)

pickle.dump(allWords, open('allWords.pickle', 'wb'))

#%%
#Hitung IDF tiap kata
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
        words_idf[word] = math.log2((N + 1)/nx)

    return words_idf

allWords_idf = computeIDF(allWords)

pickle.dump(allWords, open('allWords_idf', 'wb'))
#%%
#Buat dataframe dengan semua fitur
df = pd.DataFrame()
zeros = [0.0] * 30
for i in allWords_idf.keys():
    df[i] = zeros
    

#Isi dataframe dengan TF-IDF
for i in range(len(df.index)):
    for word in allWords_idf.keys():
        df['{}'.format(word)][i] = math.log10(documents[i].count(word) + 1) * allWords_idf[word]
        
pickle.dump(df, open('data_frame.pickle', 'wb'))

#%%
#Normalisasi Vektor
vector_length = list()
for row in df.index:
    v_len = 0
    for col in df.loc[row]:
        v_len += col
    vector_length.append(math.sqrt(v_len))