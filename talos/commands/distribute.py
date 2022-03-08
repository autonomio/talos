from ..scan.Scan import Scan

import json
import paramiko
import os
import threading
import datetime
import pandas as pd


class DistributeScan(Scan):
    def __init__(
        self,
        params,
        config="config.json",
        file_path="script.py",
        destination_path="./temp.py",
        experiment_name="talos_experiment",
    ):
        """


        Parameters
        ----------
        params : TYPE: `dict`
            DESCRIPTION. Hyperparameters for distribution.
        config : TYPE, optional
            DESCRIPTION. The default is 'config.json'.
        file_path : TYPE, optional
            DESCRIPTION. The default is 'script.py'.
        destination_path : TYPE, optional
            DESCRIPTION. The default is "./temp.py".
        experiment_name : TYPE, optional
            DESCRIPTION. The default is "talos_experiment".

        Returns
        -------
        None.

        """
        # distributed configurations
        self.params = params
        self.config = config
        self.file_path = file_path
        self.destination_path = destination_path
        self.experiment_name = experiment_name
        self.dest_dir = (
            os.path.dirname(self.destination_path) + "/" + self.experiment_name
        )
        self.save_timestamp = int(datetime.datetime.now().timestamp())

    def load_config(self):
        """
        load the machine configurations from a json file
        or directly from a dictionary.
        """
        config = self.config
        if type(config) == str:
            with open(config, "r") as f:
                data = json.load(f)
            return data["machines"]
        elif type(config) == dict:
            return config["machines"]
        else:
            TypeError("""Enter config path or config dict""")

    def create_param_space(self, n_splits=2):
        """


        Parameters
        ----------
        n_splits : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        TYPE: `list` of `dict`
            DESCRIPTION. Split parameter spaces for each machine to run.

        """

        from ..parameters.ParamSpace import ParamSpace

        params = self.params
        param_keys = params.keys()
        param_grid = ParamSpace(params, param_keys)._param_space_creation()

        def __column(matrix, i):
            return [row[i] for row in matrix]

        new_params = {k: [] for k in param_keys}
        for key_index, key in enumerate(param_keys):
            new_params[key] = __column(param_grid, key_index)

        def __split_params(n_splits=n_splits):
            d = new_params
            dicts = [{} for i in range(n_splits)]

            def _chunkify(lst, n):
                return [lst[i::n] for i in range(n)]

            for k, v in d.items():
                for i in range(n_splits):
                    dicts[i][k] = _chunkify(v, n_splits)[i]
            return dicts

        new_params = __split_params(n_splits)
        return new_params

    def ssh_connect(self):
        """


        Returns
        -------
        clients : TYPE : `list`
            DESCRIPTION. List of client objects of machines after connection.

        """
        configs = self.load_config()
        clients = []
        for config in configs:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            host = config["TALOS_IP_ADDRESS"]
            port = config["TALOS_PORT"]
            username = config["TALOS_USER"]
            if "TALOS_PASSWORD" in config.keys():
                password = config["TALOS_PASSWORD"]
                client.connect(host, port, username, password)
            elif "TALOS_KEY_FILENAME" in config.keys():
                client.connect(
                    host,
                    port,
                    username,
                    key_filename=config["TALOS_KEY_FILENAME"]
                )

            clients.append(client)
        return clients

    def ssh_run(self, client, params, machine_id):
        """


        Parameters
        ----------
        client : TYPE: `Object`
            DESCRIPTION. paramiko ssh client object
        params : TYPE: `dict`
            DESCRIPTION. hyperparameter options
        machine_id : TYPE: `int`
            DESCRIPTION. Machine id for each of the distribution machines

        Returns
        -------
        None.

        """
        sftp = client.open_sftp()
        sftp.put(self.file_path, "{}".format(self.destination_path))
        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdin, stdout, stderr = client.exec_command(
            'python3 {} "{}" {}'.format(
                self.destination_path, params, self.dest_dir)
        )
        if stderr:
            for line in stderr:
                # Process each error  line in the remote output
                print(line)

        for line in stdout:
            # Process each error  line in the remote output
            print(line)

        # fetch the latest csv
        localpath = self.experiment_name

        remotepath = self.dest_dir

        sftp.chdir(remotepath)
        for f in sorted(sftp.listdir_attr(),
                        key=lambda k: k.st_mtime,
                        reverse=True):
            sftp.get(
                f.filename,
                localpath
                + "/"
                + str(self.save_timestamp)
                + "_machine_"
                + str(machine_id)
                + ".csv",
            )
            break

        sftp.close()
        client.close()

    def run_local(self, params):
        """


        Parameters
        ----------
        params : TYPE: `dict`
            DESCRIPTION. hyperparameter options

        Returns
        -------
        None.

        """
        os.system('python3 {} "{}" {} '.format(
            self.file_path, params, self.dest_dir))

    def merge_csvs(self):
        """


        Returns
        -------
        results : TYPE: `DataFrame`
            DESCRIPTION. Returns a pandas dataframe
            after merging results from multiple machines.

        """
        localpath = self.experiment_name
        filepaths = [
            os.path.join(localpath, i)
            for i in os.listdir(localpath)
            if i.startswith(str(self.save_timestamp))
        ]
        dfs = [pd.read_csv(f) for f in filepaths]
        results = pd.concat(dfs)
        results.to_csv(
            os.path.join(localpath, str(self.save_timestamp) + "_results.csv")
        )
        return results

    def update_db(self, data_frame, config=None):
        """


        Parameters
        ----------
        data_frame : TYPE: `DataFrame`
            DESCRIPTION. The experiment results as a dataframe object
        config : TYPE:dict, optional
            DESCRIPTION. The default is None. The config is for the
                          database credentials  in case of
                          usage of a postgres or mysql engine.

        Returns
        -------
        db : TYPE: `Object`
            DESCRIPTION. A database object

        """
        from ..database.database import Database

        if config:
            username = config["DB_USERNAME"]
            password = config["DB_PASSWORD"]
            host = config["DB_HOST"]
            port = config["DB_PORT"]
            db = Database(username, password, host, port)
        else:
            db = Database()
        db.write_to_db(data_frame)
        return db

    def distributed_run(
        self,
        run_local=False,
        db_machine_id=0,
        db_config=None,
        show_results=False
    ):
        """


        Parameters
        ----------
        run_local : TYPE: `bool`, optional
            DESCRIPTION. The default is False.
        db_machine_id:TYPE : `int`
            DESCRIPTION. The default is 0. Indicates the centralised store
                          where the data gets merged.
        db_config : TYPE: `dict`, optional
            DESCRIPTION. The default is None.
        show_results : TYPE: `bool`, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        clients = self.ssh_connect()
        n_splits = len(clients)
        threads = []

        if run_local:
            n_splits += 1
            params_dict = self.create_param_space(n_splits=n_splits)
            params = params_dict[0]
            t = threading.Thread(target=self.run_local, args=(params,))
            t.start()
            threads.append(t)
            params_dict = params_dict[1:]
        else:
            params_dict = self.create_param_space(n_splits=n_splits)

        for machine_id, client in enumerate(clients):
            t = threading.Thread(
                target=self.ssh_run,
                args=(client, params_dict[machine_id], machine_id + 1),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        results = self.merge_csvs()
        db = self.update_db(results, db_config)
        if show_results:
            print(db.show_table_content())
