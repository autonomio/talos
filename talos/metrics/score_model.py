from .performance import Performance
from numpy import nan


def get_score(self):

    '''SCORE MODEL CONFIGURATION

    Initiates the model scoring process. Currently works
    only for binary and categorical predictions.

    TODO: the exception handling needs to become metric based.

    '''

    try:
        y_pred = self.keras_model.predict_classes(self.x_val)
        return Performance(y_pred, self.y_val, self.shape, self.y_max).result

    except TypeError:
        return nan
