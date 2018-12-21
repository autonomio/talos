from numpy import mean, std

from sklearn.metrics import f1_score

from ..utils.validation_split import kfold
from ..utils.best_model import best_model, activate_model


class Evaluate:

    '''Class for evaluating models based on the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object'''

        self.scan_object = scan_object
        self.data = scan_object.data

    def evaluate(self, x, y,
                 model_id=None,
                 folds=5,
                 shuffle=True,
                 average='binary',
                 metric='val_acc',
                 asc=False, 
                 print_out=False):

        '''Evaluate model against f1-score'''

        out = []
        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)

        model = activate_model(self.scan_object, model_id)

        kx, ky = kfold(x, y, folds, shuffle)

        for i in range(folds):
            y_pred = model.predict(kx[i]) >= 0.5
            scores = f1_score(y_pred, ky[i], average=average)
            out.append(scores * 100)

        if print_out is True:
            print("%.2f%% (+/- %.2f%%)" % (mean(out), std(out)))

        return out