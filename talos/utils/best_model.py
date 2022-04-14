def best_model(self, metric, asc):

    '''Picks the best model based on a given metric and
    returns the index number for the model.

    NOTE: for loss 'asc' should be True'''

    best = self.data.sort_values(metric, ascending=asc).iloc[0].name

    return best


def activate_model(self, model_id, saved=False, custom_objects=None):

    '''Loads the model from the json that is stored in the Scan object
    or from local

    model_id | int | the sequential id of the model
    saved | bool | if a model saved on local machine should be used
    custom_object | dict | if the model has a custom object, pass it here

    '''

    import tensorflow as tf
    from tensorflow.keras.models import model_from_json

    if saved:

        file_path = self.details['experiment_name']
        file_path += '/' + self.details['experiment_id']
        file_path += '/' + str(model_id)

        model = tf.keras.models.load_model(file_path,
                                           custom_objects=custom_objects)

    else:
        model = model_from_json(self.saved_models[model_id])
        model.set_weights(self.saved_weights[model_id])

    return model
