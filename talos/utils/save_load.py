from keras.models import model_from_json


def save_model(self):

    model_json = self.keras_model.to_json()
    with open(self.experiment_name + ".json", "w") as json_file:
        json_file.write(model_json)

    self.keras_model.save_weights(self.experiment_name + ".h5")


def load_model(experiment_name):

    # load json and create model
    json_file = open(experiment_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    return loaded_model.load_weights(experiment_name + ".h5")
