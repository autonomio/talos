def initialize_log(self):

    import time
    import os

    # create the experiment folder (unless one is already there)
    try:
        path = os.getcwd()
        os.mkdir(path + '/' + self.experiment_name)
    except FileExistsError:
        pass

    self._experiment_id = time.strftime('%D%H%M%S').replace('/', '')
    _file_name = self._experiment_id + '.csv'
    _experiment_log = './' + self.experiment_name + '/' + _file_name

    f = open(_experiment_log, 'w')
    f.write('')
    f.close()

    return _experiment_log
