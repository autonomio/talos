import json
import paramiko
import os
import pandas as pd


def return_current_machine_id(self,):
    ''' return machine id after checking the ip from config'''

    current_machine_id = 0
    if 'current_machine_id' in self.config_data.keys():
        current_machine_id = int(self.config_data['current_machine_id'])

    return current_machine_id


def return_central_machine_id(self):
    ''' return central machine id as mentioned in config'''
    central_id = 0
    config_data = self.config_data
    if 'database' in config_data.keys():
        central_id = int(config_data['database']['DB_HOST_MACHINE_ID'])
    return central_id


def read_config(self):
    '''read config from file'''
    with open('config.json', 'r') as f:
        config_data = json.load(f)
    return config_data


def write_config(self, new_config):
    ''' write config to file'''
    with open('config.json', 'w') as outfile:
        json.dump(new_config, outfile, indent=2)


def ssh_connect(self):
    '''
    Returns
    -------
    clients | `list` | List of client objects of machines after connection.

    '''
    configs = self.config_data['machines']
    clients = {}
    for config in configs:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        host = config['TALOS_IP_ADDRESS']
        port = config['TALOS_PORT']
        username = config['TALOS_USER']
        if 'TALOS_PASSWORD' in config.keys():
            password = config['TALOS_PASSWORD']
            client.connect(host, port, username, password)
        elif 'TALOS_KEY_FILENAME' in config.keys():
            client.connect(
                host, port, username, key_filename=config['TALOS_KEY_FILENAME']
            )

        clients[config['machine_id']] = client
    return clients


def ssh_file_transfer(self, client, machine_id):
    '''transfer the current talos script to the remote machines'''

    sftp = client.open_sftp()
    sftp.put(self.file_path, self.destination_path)
    sftp.put('./remote_config.json', self.dest_dir + '/remote_config.json')
    sftp.close()


def ssh_run(self, client, machine_id):
    '''

    Parameters
    ----------
    client | `Object` | paramiko ssh client object
    params | `dict`| hyperparameter options
    machine_id | `int`| Machine id for each of the distribution machines

    Returns
    -------
    None.

    '''

    # Run the transmitted script remotely without args and show its output.
    # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)'''

    stdin, stdout, stderr = client.exec_command(
        'python3 {}'.format(self.destination_path)
    )
    if stderr:
        for line in stderr:
            try:
                # Process each error  line in the remote output
                print(line)
            except:
                print('Cannot Output error')

    for line in stdout:
        try:
            # Process each  line in the remote output
            print(line)
        except:
            print('Cannot Output error')


def fetch_latest_file(self):

    # fetch the latest csv for an experiment'''

    experiment_name = self.experiment_name
    save_timestamp = self.save_timestamp

    if not os.path.exists(experiment_name):
        return []

    filelist = [
        os.path.join(experiment_name, i)
        for i in os.listdir(experiment_name)
        if i.endswith('.csv') and int(i.replace('.csv', '')) >= int(save_timestamp)
    ]

    if filelist:

        latest_filepath = max(filelist, key=os.path.getmtime)

        try:
            results_data = pd.read_csv(latest_filepath)
        except Exception as e:
            print('File empty..waiting...')
            return []

        return results_data

    else:
        return []


def add_experiment_id(self, results_data, machine_id):

    # create hashmap for a dataframe and use it for experiment id'''

    results_data = results_data.drop(['experiment_id'], axis=1, errors='ignore')
    results_data['experiment_id'] = pd.util.hash_pandas_object(results_data)
    results_data['experiment_id'] = [
        str(i) + '_machine_id_' + str(machine_id)
        for i in results_data['experiment_id']
    ]
    return results_data
