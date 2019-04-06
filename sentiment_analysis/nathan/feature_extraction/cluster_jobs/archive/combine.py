import numpy as np
import pandas as pd

c1 = np.load('train10k__.npy')
c2 = np.load('train10k-20k.npy')
c3 = np.load('train20k-30k.npy')
c4 = np.load('train30k-40k.npy')
c5 = np.load('train40k-50k.npy')
c6 = np.load('train50k-60k.npy')
c7 = np.load('train60k-70k.npy')
c8 = np.load('train70k-80k.npy')
c9 = np.load('train80k-90k.npy')
c10 = np.load('train90k-100k.npy')

df = pd.read_csv('./data/train.csv')

temp = np.concatenate((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10), axis=0)
print('temp shape:', temp.shape)
print('train shape before:', df.shape)
df['words'] = temp
print('train shape after:', df.shape)

df.to_csv('./data/train_CLEAN.csv', index=False)

test = pd.read_csv('./data/train_CLEAN.csv')
print('after saving train shape:', test.shape)
