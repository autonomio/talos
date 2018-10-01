from numpy import mean, std

from keras.models import model_from_json
from .validation_split import kfold
from sklearn.metrics import f1_score

class Predict:

    '''Class for making predictions on the models that are stored
    in the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object'''

        self.scan_object = scan_object
        self.data = self._dataframe()

    def _dataframe(self):

        '''Helper function to convert the scan.result to a dataframe'''

        data = self.scan_object.result[self.scan_object.result.columns[0]].str.split(',', expand=True)
        data.columns = self.scan_object.result.columns[0].split(',')

        return data

    def load_model(self, model_id):

        '''Loads the model from the json that is stored in the Scan object'''

        model = model_from_json(self.scan_object.saved_models[model_id])
        model.set_weights(self.scan_object.saved_weights[model_id])

        return model

    def best_model(self, metric='val_acc', asc=False):

        '''Picks the best model based on a given metric and
        returns the index number for the model.

        NOTE: for loss 'asc' should be True'''

        best = self.data.sort_values(metric, ascending=asc).iloc[0].name

        return best - 1

    def predict(self, x, model_id=None):

        '''Makes a probability prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            model_id = self.best_model()

        model = self.load_model(model_id)

        return model.predict(x)

    def predict_classes(self, x, model_id=None):

        '''Makes a class prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            model_id = self.best_model()

        model = self.load_model(model_id)

        return model.predict_classes(x)

    def evaluate(self, x, y,
                 model_id=None,
                 folds=5,
                 shuffle=True,
                 average='binary'):

        '''Evaluate model against f1-score'''

        out = []
        if model_id is None:
            model_id = self.best_model()

        model = self.load_model(model_id)

        kx, ky = kfold(x, y, folds, shuffle)

        for i in range(folds):
            y_pred = model.predict(kx[i]) >= 0.5
            scores = f1_score(y_pred, ky[i], average=average)
            out.append(scores * 100)

        print("%.2f%% (+/- %.2f%%)" % (mean(out), std(out)))
