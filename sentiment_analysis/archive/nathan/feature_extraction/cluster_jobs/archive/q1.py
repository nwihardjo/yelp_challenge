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
    
    for i in range(30000, 31000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())
    
    c2500 = np.array(cleaned)
    np.save('valid1000.npy', c2500)

def f2():
    global fname
    global ss
    df = pd.read_csv(fname)
    
    cleaned = []
    
    for i in range(31000, 32000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid2000.npy', c5000)
    
def f3():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(32000,33000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid3000.npy', c5000)

def f4():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(33000, 34000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid4000.npy', c5000)
    
def f5():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(34000, 35000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid5000.npy', c5000)

def f6():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(35000, 36000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid6000.npy', c5000)
    
def f7():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(36000, 37000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid7000.npy', c5000)

def f8():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(37000, 38000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid8000.npy', c5000)
    
def f9():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(38000, 39000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid9000.npy', c5000)
    
def f10():
    global fname
    global ss
    df = pd.read_csv(fname)

    cleaned = []

    for i in range(39000, 40000):
        print('cleaning index', i)
        text = ''.join(tokenize(df['text'][i]))
        suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
        cleaned.append(suggestion.split())

    c5000 = np.array(cleaned)
    np.save('valid10000.npy', c5000)
    
if __name__ == '__main__':
    p1 = multiprocessing.Process(target=f1)
    p2 = multiprocessing.Process(target=f2)
    p3 = multiprocessing.Process(target=f3)
    p4 = multiprocessing.Process(target=f4)
    p5 = multiprocessing.Process(target=f5)
    p6 = multiprocessing.Process(target=f6)
    p7 = multiprocessing.Process(target=f7)
    p8 = multiprocessing.Process(target=f8)
    p9 = multiprocessing.Process(target=f9)
    p10 = multiprocessing.Process(target=f10)
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()

    print('finished cleaning validation')
    
    c1 = np.load('valid1000.npy')
    c2 = np.load('valid2000.npy')
    c3 = np.load('valid3000.npy')
    c4 = np.load('valid4000.npy')
    c5 = np.load('valid5000.npy')
    c6 = np.load('valid6000.npy')
    c7 = np.load('valid7000.npy')
    c8 = np.load('valid8000.npy')
    c9 = np.load('valid9000.npy')
    c10 = np.load('valid10000.npy')    

    arr = np.concatenate((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10), axis=0)
    np.save('./data/train30k-40k.npy', arr)
    print(arr.shape)    
