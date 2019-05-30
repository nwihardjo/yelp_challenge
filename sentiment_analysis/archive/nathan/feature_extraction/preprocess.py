import numpy as np
import string
import os
import pandas as pd
import nltk
from symspellpy.symspellpy import SymSpell
from textblob import TextBlob
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english') + list(string.punctuation))
i = 0

def tokenize(text):
	tokens = []
	for word in nltk.word_tokenize(text):
		word = word.lower()
		if word not in stop_words and not word.isnumeric():
			tokens.append(word)
	return tokens

def clean_text(text):
	text = ''.join(tokenize(text))
	global i
	print('\t cleaning text:', i)
	i += 1
	max_edit = 2
	prefix_length = 7
	ss = SymSpell(max_edit, prefix_length)
	dict_path = "frequency_dictionary_en_82_765_GLOVE.txt"
	if not ss.load_dictionary(dict_path, 0, 1):
		print("DEBUG: DICTIONARY FILE NOT FOUND!")
	suggestion = (ss.lookup_compound((ss.word_segmentation(text)).corrected_string, max_edit))[0].term
	return suggestion.split()

if __name__ == '__main__':
	files = ['train.csv', 'test.csv', 'valid.csv']
	for file in files:
		print('Reading data from', file, '...')
		df = pd.read_csv('./data/'+file)
		df['words'] = df['text'].apply(clean_text)
		print('Writing dataframe of', file, '...')
		df.to_csv('./data/CLEANED_'+file)
