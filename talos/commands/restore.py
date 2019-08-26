class Restore:

    '''Restores the scan_object that had been stored locally as a result
    of talos.Deploy(scan_object, 'example')

    USE:

    diabetes = ta.Scan(x, y, p, input_model)
    ta.Deploy(diabetes, 'diabetes')
    ta.Restore('diabetes.zip')

    '''

    def __init__(self, path_to_zip):

        from zipfile import ZipFile

        import pandas as pd
        import numpy as np

        # create paths
        self.path_to_zip = path_to_zip
        self.extract_to = path_to_zip.replace('.zip', '')
        self.package_name = self.extract_to.split('/')[-1]
        self.file_prefix = self.extract_to + '/' + self.package_name

        # extract the zip
        # unpack_archive(self.path_to_zip, self.extract_to)
        z = ZipFile(self.path_to_zip, mode='r')
        z.extractall(self.extract_to)

        # add params dictionary
        self.params = np.load(self.file_prefix + '_params.npy',
                              allow_pickle=True).item()

        # add experiment details
        self.details = pd.read_csv(self.file_prefix + '_details.txt',
                                   header=None)

        # add x data sample
        self.x = pd.read_csv(self.file_prefix + '_x.csv', header=None)

        # add y data sample
        self.y = pd.read_csv(self.file_prefix + '_y.csv', header=None)

        # add model
        from talos.utils.load_model import load_model
        self.model = load_model(self.file_prefix + '_model')

        # add results
        self.results = pd.read_csv(self.file_prefix + '_results.csv')
        self.results.drop('Unnamed: 0', axis=1, inplace=True)

        # clean up
        del self.extract_to, self.file_prefix
        del self.package_name, self.path_to_zip
