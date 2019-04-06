from pycorenlp import StanfordCoreNLP
import pandas as pd
import numpy as np
import multiprocessing

nlp = StanfordCoreNLP('http://localhost:9000')
nlp1 = StanfordCoreNLP('http://localhost:9001')
nlp2 = StanfordCoreNLP('http://localhost:9002')
nlp3 = StanfordCoreNLP('http://localhost:9003')
nlp4 = StanfordCoreNLP('http://localhost:9004')
	
props = {'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 50000,}
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
	
def annotate1(text):
	global count
	global nlp1
	global props
	count += 1
	print('annotating:', count)

	dim = 10
	ret = np.zeros(dim)
	
	res = nlp1.annotate(text, properties=props)
	
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

def annotate2(text):
	global count
	global nlp2
	global props
	count += 1
	print('annotating:', count)

	dim = 10
	ret = np.zeros(dim)
	
	res = nlp2.annotate(text, properties=props)
	
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
 
def annotate3(text):
	global count
	global nlp3
	global props
	count += 1
	print('annotating:', count)

	dim = 10
	ret = np.zeros(dim)
	
	res = nlp3.annotate(text, properties=props)
	
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

def annotate4(text):
	global count
	global nlp4
	global props
	count += 1
	print('annotating:', count)

	dim = 10
	ret = np.zeros(dim)
	
	res = nlp4.annotate(text, properties=props)
	
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

fname = './data/yelp_no_intersect_EXPANDED.csv'

def f():
	df = pd.read_csv(fname)
	temp = []
	for i in range(0, 150000):
		temp.append(annotate(df['text'][i]))
	
	temp = np.array(temp)
	np.save('train0-20k.npy', temp)
	

def f1():
	df = pd.read_csv(fname)
	temp = []
	for i in range(150000, 300000):
		temp.append(annotate1(df['text'][i]))
	
	temp = np.array(temp)
	np.save('train20-40k.npy', temp)

def f2():
	df = pd.read_csv(fname)
	temp = []
	for i in range(120000, 180000):
		temp.append(annotate2(df['text'][i]))
	
	temp = np.array(temp)
	np.save('train40-60k.npy', temp)

def f3():
	df = pd.read_csv(fname)
	temp = []
	for i in range(180000, 240000):
		temp.append(annotate3(df['text'][i]))
	
	temp = np.array(temp)
	np.save('train60-80k.npy', temp)

def f4():
	df = pd.read_csv(fname)
	temp = []
	for i in range(240000, 300000):
		temp.append(annotate4(df['text'][i]))
	
	temp = np.array(temp)
	np.save('train80-100k.npy', temp)

if __name__ == "__main__":
	p = multiprocessing.Process(target=f)
	p1 = multiprocessing.Process(target=f1)
	#p2 = multiprocessing.Process(target=f2)
	#p3 = multiprocessing.Process(target=f3)
	#p4 = multiprocessing.Process(target=f4)

	p.start()
	p1.start()
	#p2.start()
	#p3.start()
	#p4.start()
	p.join()
	p1.join()
	#p2.join()
	#p3.join()
	#p4.join()

	c = np.load('train0-20k.npy')
	c1 = np.load('train20-40k.npy')
	#c2 = np.load('train40-60k.npy')
	#c3 = np.load('train60-80k.npy')
	#c4 = np.load('train80-100k.npy')

	final = np.concatenate((c, c1), axis=0)
	final = np.mean(final, axis=1)
	df = pd.read_csv('./data/yelp_no_intersect_EXPANDED.csv')
	df['sentiment'] = final
	df.to_csv('./data/train_v8_no_intersect.csv', index=False)
