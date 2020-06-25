def ingest_model(self):

    '''Ingests the model that is input by the user
    through Scan() model paramater.'''

    if self.sample_weight_train is not None:
        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.round_params,
                          self.sample_weight_train)

    return self.model(self.x_train,
                      self.y_train,
                      self.x_val,
                      self.y_val,
                      self.round_params)
