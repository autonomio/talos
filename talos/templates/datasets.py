def telco_churn(quantile=.5):

    '''Returns dataset in format x, [y1, y2]. This dataset
    is useful for demonstrating multi-output model or for
    experimenting with reduction strategy creation.

    The data is from hyperparameter optimization experiment with
    Kaggle telco churn dataset.

    x: features
    y1: val_loss
    y2: val_f1score

    quantile is for transforming the otherwise continuous y variables into
    labels so that higher value is stronger. If set to 0 then original
    continuous will be returned.'''

    import wrangle
    import pandas as pd

    df = pd.read_csv('https://raw.githubusercontent.com/autonomio/examples/master/telco_churn/telco_churn_for_sensitivity.csv')

    df = df.drop(['val_acc', 'loss', 'f1score', 'acc', 'round_epochs'], 1)

    for col in df.iloc[:, 2:].columns:
        df = wrangle.col_to_multilabel(df, col)

    df = wrangle.df_rename_cols(df)

    if quantile > 0:
        y1 = (df.C0 < df.C0.quantile(quantile)).astype(int).values
        y2 = (df.C1 > df.C1.quantile(quantile)).astype(int).values
    else:
        y1 = df.C0.values
        y2 = df.C1.values

    x = df.drop(['C0', 'C1'], 1).values

    return x, [y1, y2]


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

    x = x.values

    print('BE CAREFUL, this dataset has nan values.')

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


def mnist():

    '''Note that this dataset, unlike other Talos datasets,returns:

    x_train, y_train, x_val, y_val'''

    import keras
    import numpy as np

    # the data, split between train and test sets
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 28, 28

    if keras.backend.image_data_format() == 'channels_first':

        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255

    classes = len(np.unique(y_train))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, classes)
    y_val = keras.utils.to_categorical(y_val, classes)

    print("Use input_shape %s" % str(input_shape))

    return x_train, y_train, x_val, y_val
