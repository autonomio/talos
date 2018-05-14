from .performance import Performance


def get_score(self):

    y_pred = self.keras_model.predict_classes(self.x_val)
    return Performance(y_pred, self.y_val, self.shape, self.y_max).result
