from tensorflow.keras.models import model_from_json


def best_model(self, metric, asc):

    '''Picks the best model based on a given metric and
    returns the index number for the model.

    NOTE: for loss 'asc' should be True'''

    best = self.data.sort_values(metric, ascending=asc).iloc[0].name

    return best


def activate_model(self, model_id):

    '''Loads the model from the json that is stored in the Scan object'''

    model = model_from_json(self.saved_models[model_id])
    model.set_weights(self.saved_weights[model_id])

    return model
