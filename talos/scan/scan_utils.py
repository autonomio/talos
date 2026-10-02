"""Initialize legacy scan result files and model artifact directories."""


def initialize_log(self):

    import os
    import time

    # create the experiment folder (unless one is already there)
    try:
        path = os.getcwd()
        os.mkdir(path + '/' + self.experiment_name)
    except FileExistsError:
        pass

    # create unique experiment_id
    self._experiment_id = time.strftime('%D%H%M%S').replace('/', '')

    # place saved models on a sub-folder
    if self.save_models:
        self._saved_models_path = self.experiment_name + '/' + self._experiment_id
        file_path = path + '/' + self._saved_models_path
        os.mkdir(file_path)

    _experiment_log = './' + self.experiment_name + '/' + self._experiment_id + '.csv'

    f = open(_experiment_log, 'w')
    f.write('')
    f.close()

    return _experiment_log
