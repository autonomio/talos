class Predict:

    '''Class for making predictions on the models that are stored
    in the Scan() object'''

    def __init__(self, scan_object, task):

        '''Takes in as input a Scan() object'''

        self.scan_object = scan_object
        self.data = scan_object.data
        self.task = task

    def predict(self, x, model_id=None, metric='val_acc', asc=False):

        '''Makes a probability prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            from ..utils.best_model import best_model
            model_id = best_model(self.scan_object, metric, asc)

        from ..utils.best_model import activate_model
        model = activate_model(self.scan_object, model_id)

        return model.predict(x)

    def predict_classes(self, x, model_id=None, metric='val_acc', asc=False):

        '''Makes a class prediction from input x. If model_id
        is not given, then best_model will be used.'''

        if model_id is None:
            from ..utils.best_model import best_model
            model_id = best_model(self.scan_object, metric, asc)

        from ..utils.best_model import activate_model
        model = activate_model(self.scan_object, model_id)

        return model.predict_classes(x)
