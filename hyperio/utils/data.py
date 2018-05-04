import pandas as pd

def iris():

	'''
	Returns (x, y)

	'''

	df = pd.read_csv('iris.csv')
	df['class'] = df['class'].factorize()[0]
	df = df.sample(len(df))
	y = to_categorical(df['class'])
	x = df.iloc[:,:-1].values

	y = to_categorical(df['class'])
	x = df.iloc[:,:-1].values

	return x, y