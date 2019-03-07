def icu_mortality(samples=None):

    import pandas as pd
    base = 'https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/'
    df = pd.read_csv(base + 'icu_mortality.csv')
    df = df.dropna(thresh=3580, axis=1)
    df = df.dropna()
    df = df.sample(frac=1).head(samples)
    y = df['hospitalmortality'].astype(int).values
    x = df.drop('hospitalmortality', axis=1).values

    return x, y


def titanic():

    import pandas as pd
    base = 'https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/'
    df = pd.read_csv(base + 'titanic.csv')

    y = df.survived.values

    x = df[['age', 'sibsp', 'parch']]
    cols = ['class', 'embark_town', 'who', 'deck', 'sex']

    for col in cols:
        x = pd.merge(x,
                     pd.get_dummies(df[col]),
                     left_index=True,
                     right_index=True)

    x = x.dropna()
    x = x.values

    return x, y


def iris():

    import pandas as pd
    from keras.utils import to_categorical
    base = 'https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/'
    df = pd.read_csv(base + 'iris.csv')
    df['species'] = df['species'].factorize()[0]
    df = df.sample(len(df))
    y = to_categorical(df['species'])
    x = df.iloc[:, :-1].values

    y = to_categorical(df['species'])
    x = df.iloc[:, :-1].values

    return x, y


def cervical_cancer():

    import pandas as pd
    from numpy import nan
    base = 'https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/'
    df = pd.read_csv(base + 'cervical_cancer.csv')
    df = df.replace('?', nan)
    df = df.drop(['citology', 'hinselmann', 'biopsy'], axis=1)
    df = df.drop(['since_first_diagnosis',
                  'since_last_diagnosis'], axis=1).dropna()

    df = df.astype(float)

    y = df.schiller.values
    x = df.drop('schiller', axis=1).values

    return x, y


def breast_cancer():

    import pandas as pd
    base = 'https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/'
    df = pd.read_csv(base + 'breast_cancer.csv')

    # then some minimal data cleanup
    df.drop("Unnamed: 32", axis=1, inplace=True)
    df.drop("id", axis=1, inplace=True)

    # separate to x and y
    y = df.diagnosis.values
    x = df.drop('diagnosis', axis=1).values

    # convert the string labels to binary
    y = (y == 'M').astype(int)

    return x, y
