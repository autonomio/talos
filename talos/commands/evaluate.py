import keras

from numpy import mean, std

from sklearn.metrics import f1_score

from ..utils.validation_split import kfold
from ..utils.best_model import best_model, activate_model


class Evaluate:

    '''Class for evaluating models based on the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object.
        e = evaluate(scan_object) and see docstring 
        for e() for more information.'''

        self.scan_object = scan_object
        self.data = scan_object.data

    def evaluate(self, x, y,
                 model_id=None,
                 folds=5,
                 shuffle=True,
                 average='binary',
                 metric='val_acc',
                 eval_metric='f1_score',
                 eval_backend='sklearn',
                 asc=False, 
                 print_out=False):

        '''Evaluates the model based on keras or sklearn 
        metrics. By default set to f1_score but any sklearn
        or keras metric can be used. Note that 'metric' 
        is referring to the metric in scan_object.data and
        is used to first identify the best model which is then
        evaluated using eval_metric.

        SKLEARN/KERAS METRIC WITHOUT ARGUMENTS:

            evaluate(x, y,
                     eval_metric=some_sklearn_metric,
                     eval_backend='sklearn')


        SKLEARN/KERAS METRIC WITH ARGUMENTS:

            evaluate(x, y,
                     eval_metric=some_sklearn_metric(x, y, some_arg),
                     eval_backend='custom')


        CUSTOM METRIC:

            evaluate(x, y,
                     eval_metric=custom_metric(x, y, some_arg),
                     eval_backend='custom')

        x : array
            The input data for making predictions
        y : array
            The ground truth for x
        model_id : int
            It's possible to evaluate a specific model based on ID. Can be None.
        folds : int
            Number of folds to use for cross-validation
        sort_metric : string
            A column name referring to the metric that was used in the scan_object
            as a performance metric. This is used for sorting the results to pick 

        shuffle : bool
            Data is shuffled before evaluation.
        average : string
            For f1_score metric from sklearn
        evaluation_metric : string or function
            This should be one of sklearn or keras available metrics. Allows custom
            metrics to be input as functions. Note that custom function require
            evaluation_backend to be 'custom'
        evaluation_backend : string
            'keras', 'sklearn', or 'custom'
        asc : bool
            False if the metric is to be optimized upwards (e.g. accuracy or f1_score)
        print_out : bool
            Print out the results. 

        TODO: add possibility to input custom metrics.

        '''

        out = []
        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)

        model = activate_model(self.scan_object, model_id)

        kx, ky = kfold(x, y, folds, shuffle)

        for i in range(folds):
            y_pred = model.predict(kx[i]) >= 0.5

            elif eval_metric == 'f1_score':
                scores = f1_score(y_pred, ky[i], average=average)

            else:
                if eval_backend == 'keras'
                    scores = keras.metrics.__getattribute__(eval_metric)(y_pred, ky[i])
                
                elif eval_backend == 'sklearn':
                    scores = sklearn.metrics.__getattribute__(eval_metric)(y_pred, ky[i])

                elif eval_backend == 'custom'
                    scores = eval_metric

            out.append(scores * 100)

        if print_out is True:
            print("%.2f%% (+/- %.2f%%)" % (mean(out), std(out)))

        return out