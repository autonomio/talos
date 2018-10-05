from numpy import mean, std

from sklearn.metrics import f1_score

from ..utils.validation_split import kfold
from ..utils.best_model import best_model, activate_model


class Predict:

    '''Class for making predictions on the models that are stored
    in the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object'''

        self.scan_object = scan_object
        self.data = scan_object.data

    def predict(self, x, model_id=None, metric='val_acc', asc=False):

        '''Makes a probability prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)

        model = activate_model(self.scan_object, model_id)

        return model.predict(x)

    def predict_classes(self, x, model_id=None, metric='val_acc', asc=False):

        '''Makes a class prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)

        model = activate_model(self.scan_object, model_id)

        return model.predict_classes(x)
