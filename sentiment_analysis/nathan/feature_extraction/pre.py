import numpy as np
import string
import os
import multiprocessing
import pandas as pd
import nltk
from symspellpy.symspellpy import SymSpell
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english') + list(string.punctuation))

def tokenize(text):
        tokens = []
        for word in nltk.word_tokenize(text):
                word = word.lower()
                if word not in stop_words and not word.isnumeric():
                        tokens.append(word)
        return tokens

fname = './data/train.csv'    
dirr = './data/'
max_edit = 2
prefix_length = 7
ss = SymSpell(max_edit, prefix_length)
dict_path = "frequency_dictionary_en_82_765_GLOVE.txt"
if not ss.load_dictionary(dict_path, 0, 1):
        print("DEBUG: DICTIONARY FILE NOT FOUND!")

def f1():
    global fname
    global ss
    df = pd.read_csv(fname)
    
    cleaned = []
    
    for i in range(10000, 12500):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())
    
    c2500 = np.array(cleaned)
    np.save(dirr+'train2500.npy', c2500)

def f2():
    global fname
    global ss
    df = pd.read_csv(fname)
    
    cleaned = []
    
    for i in range(12500, 15000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save(dirr+'train5000.npy', c5000)

def f3():
    global fname
    global ss
    df = pd.read_csv(fname)
    
    cleaned = []
    
    for i in range(17500, 20000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c10k = np.array(cleaned)
    np.save(dirr+'train10k.npy', c10k)

def f4():
    global fname
    global ss
    df = pd.read_csv(fname)
    
    cleaned = []
    
    for i in range(15000, 17500):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c7_5k = np.array(cleaned)
    np.save(dirr+'train7500.npy', c7_5k)

    
if __name__ == '__main__':
    p1 = multiprocessing.Process(target=f1)
    p2 = multiprocessing.Process(target=f2)
    p3 = multiprocessing.Process(target=f3)
    p4 = multiprocessing.Process(target=f4)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    print('Finished execution')

    c1 = np.load(dirr+'train2500.npy')
    c2 = np.load(dirr+'train5000.npy')
    c3 = np.load(dirr+'train7500.npy')
    c4 = np.load(dirr+'train10k.npy')
    arr = np.concatenate((c1,c2,c3,c4),axis=0)
    np.save(dirr+'train10k-20k.npy', arr)
