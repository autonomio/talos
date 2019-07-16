class Deploy:

    '''Functionality for deploying a model to a filename'''

    def __init__(self, scan_object, model_name, metric, asc=False):

        '''Deploy a model to be used later or in a different system.

        NOTE: for a metric that is to be minimized, set asc=True or otherwise
        you will end up with the model that has the highest loss.

        Deploy() takes in the object from Scan() and creates a package locally
        that can be later activated with Restore().

        scan_object : object
            The object that is returned from Scan() upon completion.
        model_name : str
            Name for the .zip file to be created.
        metric : str
            The metric to be used for picking the best model.
        asc: bool
            Make this True for metrics that are to be minimized (e.g. loss) ,
            and False when the metric is to be maximized (e.g. acc)

        '''

        import os

        self.scan_object = scan_object
        os.mkdir(model_name)
        self.path = model_name + '/' + model_name
        self.model_name = model_name
        self.metric = metric
        self.asc = asc
        self.data = scan_object.data

        from ..utils.best_model import best_model, activate_model
        self.best_model = best_model(scan_object, metric, asc)
        self.model = activate_model(scan_object, self.best_model)

        # runtime
        self.save_model_as()
        self.save_details()
        self.save_data()
        self.save_results()
        self.save_params()
        self.save_readme()
        self.package()

    def save_model_as(self):

        '''Model Saver
        WHAT: Saves a trained model so it can be loaded later
        for predictions by predictor().
        '''

        model_json = self.model.to_json()
        with open(self.path + "_model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(self.path + "_model.h5")
        print("Deploy package" + " " + self.model_name + " " + "have been saved.")

    def save_details(self):

        self.scan_object.details.to_csv(self.path + '_details.txt')

    def save_data(self):

        import pandas as pd

        # input data is <= 2d
        try:
            x = pd.DataFrame(self.scan_object.x[:100])
            y = pd.DataFrame(self.scan_object.y[:100])

        # input data is > 2d
        except ValueError:
            x = pd.DataFrame()
            y = pd.DataFrame()
            print("data is not 2d, dummy data written instead.")

        x.to_csv(self.path + '_x.csv', header=None, index=None)
        y.to_csv(self.path + '_y.csv', header=None, index=None)

    def save_results(self):

        self.scan_object.data.to_csv(self.path + '_results.csv')

    def save_params(self):

        import numpy as np

        np.save(self.path + '_params', self.scan_object.params)

    def save_readme(self):

        txt = 'To activate the assets in the Talos deploy package: \n\n   from talos.commands.restore import Restore \n   a = Restore(\'path_to_asset\')\n\nNow you will have an object similar to the Scan object, which can be used with other Talos commands as you would be able to with the Scan object'

        text_file = open(self.path.split('/')[0] + '/README.txt', "w")
        text_file.write(txt)
        text_file.close()

    def package(self):

        import shutil

        shutil.make_archive(self.model_name, 'zip', self.model_name)
        shutil.rmtree(self.model_name)
