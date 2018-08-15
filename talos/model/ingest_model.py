def ingest_model(self):

    return self.model(self.x_train,
                      self.y_train,
                      self.x_val,
                      self.y_val,
                      self.round_params)
