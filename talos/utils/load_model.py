from tensorflow.keras.models import model_from_json


def load_model(saved_model):

    '''Load a Model from local disk

    Takes as input .json and .h5 file with model
    and weights and returns a model that can be then
    used for predictions.

    saved_model :: name of the saved model without
    suffix (e.g. 'iris_model' and not 'iris_model.json')

    '''

    json_file = open(saved_model + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(saved_model + '.h5')

    return model
