from .binary_performance import BinaryPerformance


def get_score(self):

    y_pred = self.keras_model.predict_classes(self.x_val)
    return BinaryPerformance(y_pred, self.y_val).result
