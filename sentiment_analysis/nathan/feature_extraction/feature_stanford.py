from pycorenlp import StanfordCoreNLP
import pandas as pd
import numpy as np

nlp = StanfordCoreNLP('http://localhost:9000')
props = {'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 30000,}
#fname = ['./data/valid_CLEAN.csv', './data/test_CLEAN.csv', './data/train_CLEAN.csv']

count = 0

def annotate(text):
	global count
	global nlp
	global props
	count += 1
	print('annotating:', count)

	dim = 10
	ret = np.zeros(dim)
	
	res = nlp.annotate(text, properties=props)
	
	sentimentVals = np.zeros(len(res['sentences']))
	for i, s in enumerate(res['sentences'], 0):
		if i+1 > dim: 
			break
		ret[i] = s['sentimentValue']
		sentimentVals[i] = s['sentimentValue']

	if len(res['sentences']) < dim:
		sentenceLength = len(res['sentences'])
		mean = np.mean(sentimentVals)
		scale = 1
		size = dim - sentenceLength

		ret[sentenceLength:] = np.random.normal(loc=mean, scale=scale,size=size)
	
	return ret
	
if __name__ == "__main__":
	fnames = ['valid', 'test']
	dir_ = './data/'
	for fname in fnames:
		df = pd.read_csv(dir_+fname+'_CLEAN.csv')
		a = np.zeros((10000, 10))
		for i in range(0, 10000):
			a[i,:] = annotate(df['text'][i])
		np.save(fname+'.npy', a)
		
		for i in range(0, 10):
			s = 's'+str(i)
			df[s] = a[:,i]

		df.to_csv(dir_+fname+'_v3.csv', index=False)
