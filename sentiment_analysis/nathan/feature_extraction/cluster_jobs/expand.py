import numpy as np
import pandas as pd
from textblob import TextBlob

def p(text):
	return TextBlob(text).sentiment.polarity

def s(text):
	return TextBlob(text).sentiment.subjectivity

fnames = ['train_big']
dir_ = './data/'

business = pd.read_csv(dir_+'business.csv')

for fname in fnames:
	df = pd.read_csv(dir_+fname+'.csv')
	
	df['subjectivity'] = df['text'].apply(s)
	df['polarity'] = df['text'].apply(p)

	starsArr = np.zeros((df.shape[0],))
	reviewArr = np.zeros((df.shape[0],))

	for i in range(0, df.shape[0]):
		row = business.loc[business['business_id'] == df['business_id'][i]]
		starsArr = row['stars']
		reviewArr = row['review_count']
		
	df['business_stars'] = starsArr
	df['business_rcount'] = reviewArr
 
	df.to_csv(dir_+fname+'_EXPANDED.csv', index=False)
	print(df.shape)
