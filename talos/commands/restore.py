from zipfile import ZipFile

from pandas import read_csv
from numpy import load

from talos.utils.load_model import load_model


class Restore:

    '''Utility class for restoring the assets from Deploy()
    package.'''

    def __init__(self, path_to_zip):

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
        self.params = load(self.file_prefix + '_params.npy').item()

        # add experiment details
        self.details = read_csv(self.file_prefix + '_details.txt', header=None)

        # add x data sample
        self.x = read_csv(self.file_prefix + '_x.csv', header=None)

        # add y data sample
        self.y = read_csv(self.file_prefix + '_y.csv', header=None)

        # add model
        self.model = load_model(self.file_prefix + '_model')

        # add results
        self.results = read_csv(self.file_prefix + '_results.csv')
        self.results.drop('Unnamed: 0', axis=1, inplace=True)

        # clean up
        del self.extract_to, self.file_prefix
        del self.package_name, self.path_to_zip
