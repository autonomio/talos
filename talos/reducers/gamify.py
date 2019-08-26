def gamify(self):

    '''Will apply reduction changes based on edits on the
    the produced .json file in the experiment folder'''

    if self.param_object.round_counter == 1:

        # create the gamify object
        from .GamifyMap import GamifyMap
        g = GamifyMap(self)

        # keep in scan_object
        self._gamify_object = g

        # do the first export in the experiment folder
        g.export_json()

        return self

    # for every round check if there are changes
    self._gamify_object.import_json()
    self = self._gamify_object.run_updates()
    self._gamify_object.export_json()

    return self
