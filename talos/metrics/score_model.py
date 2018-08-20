from .performance import Performance
from numpy import nan


def get_score(self):

    '''SCORE MODEL CONFIGURATION

    Initiates the model scoring process. Currently works
    only for binary and categorical predictions. The model scoring
    currently is only used in the Talos master log, and is not
    available in the experiment log or reports.

    TODO: the exception handling needs to become metric based.

    '''

    try:
        y_pred = self.keras_model.predict_classes(self.x_val)
        return Performance(y_pred, self.y_val, self.shape, self.y_max).result

    except TypeError:
        return nan
