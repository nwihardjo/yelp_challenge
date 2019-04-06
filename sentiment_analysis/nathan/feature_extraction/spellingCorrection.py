import os
from symspellpy.symspellpy import SymSpell, Verbosity
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english')+list(string.punctuation))

def tokenize(text):
	tokens = []
	for word in nltk.word_tokenize(text):
		word = word.lower()
		if word not in stop_words and not word.isnumeric():
			tokens.append(word)
	return tokens

def get_data():
	df = pd.read_csv('./data/train.csv')
	return df

if __name__ == "__main__":
	ss = SymSpell(2, 7)
	dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765_GLOVE.txt")
	if not ss.load_dictionary(dict_path, 0, 1):
		print('dictionary file not found')
	
	df = get_data() 
	input = ''.join(tokenize(df['text'][80517]))
	# input = ("thequickbrowndogjumpsoverthefrog")
	max_edit = 2
	
	suggestions = ss.word_segmentation(input)
	suggestions = ss.lookup_compound(suggestions.corrected_string, 2)
	for suggestion in suggestions:
		print("{}, {}, {}".format(suggestion.term, suggestion.distance, suggestion.count))
