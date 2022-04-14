class Predict:

    '''Class for making predictions on the models that are stored
    in the Scan() object'''

    def __init__(self, scan_object):

        '''Takes in as input a Scan() object and returns and object
        with properties for `predict` and `predict_classes`'''

        self.scan_object = scan_object
        self.data = scan_object.data

    def predict(self,
                x,
                metric,
                asc,
                model_id=None,
                saved=False,
                custom_objects=None):

        '''Makes a probability prediction from input x. If model_id
        is not given, then best_model will be used.

        x | array | data to be used for the predictions
        model_id | int | the id of the model from the Scan() object
        metric | str | the metric to be used for picking best model
        asc | bool | True if `metric` is something to be minimized
        saved | bool | if a model saved on local machine should be used
        custom_objects | dict | if the model has a custom object,
                                pass it here

        '''

        if model_id is None:
            from ..utils.best_model import best_model
            model_id = best_model(self.scan_object, metric, asc)

        from ..utils.best_model import activate_model
        model = activate_model(self.scan_object,
                               model_id,
                               saved,
                               custom_objects)

        return model.predict(x)

    def predict_classes(self,
                        x,
                        metric,
                        asc,
                        model_id=None,
                        saved=False,
                        custom_objects=None):

        '''Makes a class prediction from input x. If model_id
        is not given, then best_model will be used.

        x | array | data to be used for the predictions
        model_id | int | the id of the model from the Scan() object
        metric | str | the metric to be used for picking best model
        asc | bool | True if `metric` is something to be minimized
        saved | bool | if a model saved on local machine should be used
        custom_objects | dict | if the model has a custom object, pass it here
        '''

        import numpy as np

        if model_id is None:
            from ..utils.best_model import best_model
            model_id = best_model(self.scan_object, metric, asc)

        from ..utils.best_model import activate_model
        model = activate_model(self.scan_object,
                               model_id,
                               saved,
                               custom_objects)

        # make (class) predictions with the model
        preds = model.predict(x)   
        preds_classes = np.argmax(preds, axis=1)

        return preds_classes
