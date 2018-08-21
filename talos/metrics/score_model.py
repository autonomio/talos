from .performance import Performance
from numpy import nan
from keras.models import Sequential


#   Inspired by predict_classes function from Keras Sequential Model   
#   Supports Functional Model (Experimental)
def __predict_classes(model, x, batch_size=32, verbose=0):

        proba = model.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


def get_score(self):

    '''SCORE MODEL CONFIGURATION

    Initiates the model scoring process. Currently works
    only for binary and categorical predictions.

    TODO: the exception handling needs to become metric based.

    '''

    try:
        if type(self.keras_model) is Sequential:
            y_pred = self.keras_model.predict_classes(self.x_val)
        else:
            if self.experimental_functional_support:
                y_pred = __predict_classes(self.keras_model, self.x_val)
            else:
                print("Add 'experimental_functional_support=True' in Scan() to support Functional Model")
                raise
                
        return Performance(y_pred, self.y_val, self.shape, self.y_max).result

    except TypeError:
        return nan

    
