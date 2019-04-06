import numpy as np
from symspellpy.symspellpy import SymSpell, Verbosity
import pandas as pd

if __name__ == '__main__':
	fname = './dictionary/glove.6B.50d.txt'
	dname = './dictionary/frequency_dictionary_en_82_765.txt'
	uname = './dictionary/frequency_dictionary_en_82_765_GLOVE.txt'	
	gname = './dictionary/unigram_freq.csv'
	iname = './dictionary/iweb_wordFreq_sample_lemmas.txt'

	glove_index = [] 
	existing_index = dict()
	iweb = dict()

	with open(iname, 'r') as i:
		for line in i:
			values = line.split()
			if len(values) > 0:
				iweb[values[2]] = values[4]

	df = pd.read_csv(gname)
	google_freq = dict(zip(df['word'], df['count']))
	
	with open(fname, 'r') as f:
		for line in f:	
			values = line.split()
			glove_index.append(values[0])

	with open(dname, 'r') as d:
		for line in d:
			values = line.split()
			existing_index[values[0]] = [values[1]]
			
	append_size = len(glove_index) - len(existing_index)
	fin_append = 0
	
	with open(uname, 'a') as u:
		for word, i in existing_index.items():
			u.write(word+' '+str(i)+'\n')

		for word in glove_index:
			goog = True
			if existing_index.get(word) is not None and (google_freq.get(word) is not None or iweb.get(word) is not None):
					fin_append+=1
					val = google_freq.get(word) if google_freq.get(word) is not None else iweb.get(word)
					print('Appending: '+word+' '+str(val))
					u.write(word+' '+str(val)+'\n')
	
	print("Difference between existing and glove words: ", append_size)
	print("Appended size: ", fin_append)
