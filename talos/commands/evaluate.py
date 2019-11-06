class Evaluate:

    '''Class for evaluating models based on the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object.
        e = evaluate(scan_object) and see docstring
        for e() for more information.'''

        self.scan_object = scan_object
        self.data = scan_object.data

    def evaluate(self,
                 x,
                 y,
                 task,
                 metric,
                 model_id=None,
                 folds=5,
                 shuffle=True,
                 asc=False,
                 print_out=False):

        '''Evaluate a model based on f1_score (all except regression)
        or mae (for regression). Supports 'binary', 'multi_class',
        'multi_label', and 'regression' evaluation.

        x : array
            The input data for making predictions
        y : array
            The ground truth for x
        model_id : int
            It's possible to evaluate a specific model based on ID.
            Can be None.
        folds : int
            Number of folds to use for cross-validation
        sort_metric : string
            A column name referring to the metric that was used in the
            scan_object as a performance metric. This is used for sorting
            the results to pick for evaluation.
        shuffle : bool
            Data is shuffled before evaluation.
        task : string
            'binary', 'multi_class', 'multi_label', or 'continuous'.
        asc : bool
            False if the metric is to be optimized upwards
            (e.g. accuracy or f1_score)
        print_out : bool
            Print out the results.

        TODO: add possibility to input custom metrics.

        '''

        import numpy as np
        import sklearn as sk

        out = []
        if model_id is None:
            from ..utils.best_model import best_model
            model_id = best_model(self.scan_object, metric, asc)

        from ..utils.best_model import activate_model
        model = activate_model(self.scan_object, model_id)

        from ..utils.validation_split import kfold
        kx, ky = kfold(x, y, folds, shuffle)

        for i in range(folds):

            y_pred = model.predict(kx[i], verbose=0)

            if task == 'binary':
                y_pred = np.array(y_pred) >= .5
                scores = sk.metrics.f1_score(y_pred, ky[i], average='binary')

            elif task == 'multi_class':
                y_pred = y_pred.argmax(axis=-1)
                scores = sk.metrics.f1_score(y_pred, ky[i], average='macro')

            if task == 'multi_label':
                y_pred = model.predict(kx[i]).argmax(axis=1)
                scores = sk.metrics.f1_score(y_pred,
                                             ky[i].argmax(axis=1),
                                             average='macro')

            elif task == 'continuous':
                y_pred = model.predict(kx[i])
                scores = sk.metrics.mean_absolute_error(y_pred, ky[i])

            out.append(scores)

        if print_out is True:
            print("mean : %.2f \n std : %.2f" % (np.mean(out), np.std(out)))

        return out
