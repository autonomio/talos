class GamifyMap:

    def __init__(self, scan_object):

        '''GamifyMap handles the management of the
        dictionary that contains the information about
        hyperparameters, which is exchanged in and out
        during the `Scan()` experiment.
        '''

        self.params = scan_object.param_object.params
        self.scan_object = scan_object
        self.generate_gamify_dict()
        self.gamify_map = self.generate_gamify_dict_map()

        # parse together the output file name
        _folder = './' + scan_object.experiment_name + '/'
        _id = scan_object._experiment_id
        self._filename = _folder + _id

    def run_updates(self):

        for key in self.gamify_dict.keys():
            for val in self.gamify_dict[key].keys():
                if self.gamify_dict[key][val] != self.updated_dict[key][val]:

                    label = list(self.params.keys())[int(key)]
                    value = self.params[label][int(val)]

                    self.gamify_dict[key][val] = self.updated_dict[key][val]
                    self.scan_object.param_object.remove_is(label, value)

        return self.scan_object

    def back_to_original(self, gamify_from_json):

        gamify_dict = {}
        for key in gamify_from_json:
            param_vals = {}
            for val in gamify_from_json[key]:
                param_vals[self.gamify_map[key][val][1]] = gamify_from_json[key][val][:2]
            gamify_dict[self.gamify_map[key][val][0]] = param_vals

        return gamify_dict

    def generate_gamify_dict_map(self):

        gamify_dict_map = {}

        for i, key in enumerate(self.params.keys()):
            param_vals = {}
            for ii, val in enumerate(self.params[key]):
                param_vals[str(ii)] = [key, val]
            gamify_dict_map[str(i)] = param_vals

        return gamify_dict_map

    def generate_gamify_dict(self):

        '''This is done once at the beginning
        of the experiment.

        NOTE: This will be all stringified, so an index
        mapping system will be used to convert back to
        actual forms later.'''

        gamify_dict = {}

        for i, key in enumerate(self.params.keys()):
            param_vals = {}
            for ii, val in enumerate(self.params[key]):
                param_vals[str(ii)] = ['active', 0, str(key), str(val)]
            gamify_dict[str(i)] = param_vals

        self.gamify_dict = gamify_dict

    def export_json(self):

        import json

        with open(self._filename + '.json', 'w') as fp:
            json.dump(self.gamify_dict, fp)

    def import_json(self):

        import json

        with open(self._filename + '.json', 'r') as fp:
            out = json.load(fp)

        self.updated_dict = out
