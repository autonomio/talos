import pandas as pd
from keras.utils import to_categorical


def iris():

    df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
    df['species'] = df['species'].factorize()[0]
    df = df.sample(len(df))
    y = to_categorical(df['species'])
    x = df.iloc[:, :-1].values

    y = to_categorical(df['species'])
    x = df.iloc[:, :-1].values

    return x, y
