def initialize_log(self):
    """
    Create a log file and return the file path
    """
    import time
    import os
    from re import search

    # create the experiment folder (unless one is already there)
    try:
        path = os.getcwd()
        os.mkdir(path + "/" + self.experiment_name)
    except FileExistsError:
        pass

    if self.allow_resume and any(
        (match := search(r"\.csv$", file))
        for file in os.listdir("./" + self.experiment_name)
    ):
        self._experiment_id = match.string # pylint: disable=undefined-variable
        _file_name = self._experiment_id + ".csv"
        _experiment_log = "./" + self.experiment_name + "/" + _file_name
    else:
        self._experiment_id = time.strftime("%D%H%M%S").replace("/", "")
        _file_name = self._experiment_id + ".csv"
        _experiment_log = "./" + self.experiment_name + "/" + _file_name

        f = open(_experiment_log, "w")
        f.write("")
        f.close()

    return _experiment_log


def initialize_config(self) -> str:
    """
    Construct and return a config file path
    """
    import time
    import os
    from re import search

    # create the experiment folder (unless one is already there)
    try:
        path = os.getcwd()
        os.mkdir(path + "/" + self.experiment_name)
    except FileExistsError:
        pass

    if (
        len(
            files := [
                file
                for file in os.listdir("./" + self.experiment_name)
                if search(r"(?<!keys).yaml$", file)
            ]
        )
        == 1
    ):
        _config_file = "./" + self.experiment_name + "/" + files[0]
    elif len(files) > 1:
        raise FileExistsError("Too many .yaml files found in path, abandoning.")
    else:
        self._experiment_id = time.strftime("%D%H%M%S").replace("/", "")
        _file_name = self._experiment_id + ".yaml"
        _config_file = "./" + self.experiment_name + "/" + _file_name

    return _config_file

def initialize_pickle(self) -> str:
    """
    Construct and return a pickle file path
    """
    import time
    import os
    from re import search

    # create the experiment folder (unless one is already there)
    try:
        path = os.getcwd()
        os.mkdir(path + "/" + self.experiment_name)
    except FileExistsError:
        pass

    if (
        len(
            files := [
                file
                for file in os.listdir("./" + self.experiment_name)
                if search(r".pickle$", file)
            ]
        )
        == 1
    ):
        _pickle_file = "./" + self.experiment_name + "/" + files[0]
    elif len(files) > 1:
        raise FileExistsError("Too many .pickle files found in path, abandoning.")
    else:
        self._experiment_id = time.strftime("%D%H%M%S").replace("/", "")
        _file_name = self._experiment_id + ".pickle"
        _pickle_file = "./" + self.experiment_name + "/" + _file_name

    return _pickle_file