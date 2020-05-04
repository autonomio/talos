def initialize_log(self):

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
        _experiment_log = match.string
    else:
        self._experiment_id = time.strftime("%D%H%M%S").replace("/", "")
        _file_name = self._experiment_id + ".csv"
        _experiment_log = "./" + self.experiment_name + "/" + _file_name

        f = open(_experiment_log, "w")
        f.write("")
        f.close()

    return _experiment_log


def initialize_config(self) -> str:

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
