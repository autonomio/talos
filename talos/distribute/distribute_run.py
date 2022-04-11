import json
import threading
from .distribute_params import run_scan_with_split_params
from .distribute_utils import return_current_machine_id, ssh_connect, ssh_file_transfer, ssh_run
from .distribute_database import update_db


def run_central_machine(self, n_splits, run_central_node):
    '''

    Parameters
    ----------
    params  | `dict` | hyperparameter options

    Returns
    -------
    None.

    '''

    # runs the experiment in central machine

    machine_id = 0
    run_scan_with_split_params(self, n_splits, run_central_node, machine_id)


def distribute_run(self):
    '''


    Parameters
    ----------
    run_central_machine | `bool` |The default is False.
    db_machine_id | `int` | The default is 0. Indicates the centralised store
                              where the data gets merged.

    Returns
    -------
    None.

    '''

    # run the Scan script in distributed machines

    config = self.config_data

    if 'run_central_node' in config.keys():
        run_central_node = config['run_central_node']
    else:
        run_central_node = False

    update_db_n_seconds = 5
    if 'DB_UPDATE_INTERVAL' in config['database'].keys():
        update_db_n_seconds = int(config['database']['DB_UPDATE_INTERVAL'])

    n_splits = len(config['machines'])

    if run_central_node:
        n_splits += 1

    current_machine_id = str(return_current_machine_id(self))

    if current_machine_id == str(0):

        clients = ssh_connect(self)

        for machine_id, client in clients.items():
            new_config = config
            new_config['current_machine_id'] = machine_id
            with open('tmp/remote_config.json', 'w') as outfile:
                json.dump(new_config, outfile)
            ssh_file_transfer(self, client, machine_id)
        threads = []

        if run_central_node:

            t = threading.Thread(
                target=run_central_machine,
                args=(self, n_splits, run_central_node),
            )
            t.start()
            threads.append(t)

            t = threading.Thread(
                target=update_db,
                args=([self, update_db_n_seconds, current_machine_id, self.stage]),
            )
            t.start()
            threads.append(t)

        for machine_id, client in clients.items():
            t = threading.Thread(
                target=ssh_run,
                args=(self,
                      client,
                      machine_id,
                      ),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
