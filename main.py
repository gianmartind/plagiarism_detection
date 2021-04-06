import pandas as pd
import re
import os
import glob
import operator

def cleaner(text):
    #remove newline
    text = text.replace('\n', ' ')
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    #remove special character
    text = re.sub('[^A-Za-z ]', '', text)

    return text.lower()

documents = []
for filename in glob.glob(os.path.join('dataset/', '*.txt')):
   with open(os.path.join(os.getcwd(), filename), 'r', encoding='mbcs') as f: # open in readonly mode
      documents.append(cleaner(f.read()))

def computeWordFreq(text):
    freq = len(text.split())
    wordFreq = dict()
    for i in text.split():
        if i in wordFreq.keys():
            wordFreq[i] = wordFreq[i] + 1
        else:
            wordFreq[i] = 1
    #return sorted dictionary
    sorted = sorted(wordFreq.items(), key=operator.itemgetter(0))
    return sorted

def computeAllWords(docs):
    all_words = set()
    for text in docs:
        for word in text.split():
            all_words.add(word)
    return all_words

allWords = computeAllWords(documents)

def computeIDF(allWords):
    words_idf = dict()
    for word in allWords:
        words_idf[word] = 0

    for word in allWords:
        for doc in documents:
            if word in doc:
                words_idf[word] += 1

    return words_idf

def removeStopWords(allWords):
    allWords_ = dict()
    for word in allWords.keys():
        if allWords[word] > 1:
            allWords_[word] = allWords[word]
    return allWords_

df = pd.DataFrame()
doc = ['DOC {}'.format(i) for i in range(30)]
df['doc'] = doc
zeros = [0] * 30
for i in allWords.keys():
    df[i] = zeros