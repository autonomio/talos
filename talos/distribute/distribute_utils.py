import json
import paramiko
import os
import pandas as pd


def create_temp_file(self):
    filestr = '''
from talos import RemoteScan
import numpy as np
import json
import pickle

x=np.load('tmp/x_data_remote.npy')
y=np.load('tmp/y_data_remote.npy')
    
{}

with open('tmp/arguments_remote.json','r') as f:
    arguments_dict=json.load(f)
    
t=RemoteScan(x=x,
             y=y,
             params=arguments_dict['params'],
             model={},
             experiment_name=arguments_dict['experiment_name'],
             x_val=arguments_dict['x_val'],
             y_val=arguments_dict['y_val'],
             val_split=arguments_dict['val_split'],
             random_method=arguments_dict['random_method'],
             seed=arguments_dict['seed'],
             performance_target=arguments_dict['performance_target'],
             fraction_limit=arguments_dict['fraction_limit'],
             round_limit=arguments_dict['round_limit'],
             time_limit=arguments_dict['time_limit'],
             boolean_limit=arguments_dict['boolean_limit'],
             reduction_method=arguments_dict['reduction_method'],
             reduction_interval=arguments_dict['reduction_interval'],
             reduction_window=arguments_dict['reduction_window'],
             reduction_threshold=arguments_dict['reduction_threshold'],
             reduction_metric=arguments_dict['reduction_metric'],
             minimize_loss=arguments_dict['minimize_loss'],
             disable_progress_bar=arguments_dict['disable_progress_bar'],
             print_params=arguments_dict['print_params'],
             clear_session=arguments_dict['clear_session'],
             save_weights=arguments_dict['save_weights'],
             config='tmp/remote_config.json'
             )
    '''.format(self.model_func, self.model_name)

    with open("tmp/scanfile_remote.py", "w") as f:
        f.write(filestr)


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


def read_config(self,remote=False):
    '''read config from file'''
    if remote:
        
        config_path="tmp/remote_config.json"
    else: 
        
        config_path="config.json"
        
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data


def write_config(self, new_config,remote=False):
    ''' write config to file'''
    
    if remote:
        
        config_path="tmp/remote_config.json"
        
    else: 
        
        config_path="config.json"
        
    with open(config_path, 'w') as outfile:
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
    create_temp_file(self)

    sftp = client.open_sftp()

    try:
        sftp.chdir(self.dest_dir)  # Test if dest dir exists
    except IOError:
        sftp.mkdir(self.dest_dir)  # Create dest dir
        sftp.chdir(self.dest_dir)

    for file in os.listdir("tmp"):
        sftp.put("tmp/"+file, file)

    sftp.put('tmp/remote_config.json', 'remote_config.json')
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
        'python3 tmp/scanfile_remote.py')
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
            return results_data
        except Exception as e:
            return []
    else:
        return []
    
def get_experiment_stage(self,db):
    
    try:
        ids = db.return_existing_experiment_ids()
        stage=int(list(ids)[-1].split("-")[0])+1
        
    except Exception as e:
        stage=0
        
    return stage
    

def add_experiment_id(self, results_data, machine_id,start_row,end_row,db,stage):

    # generate experiment id from model id and row number'''
    # results_data = results_data.drop(['experiment_id'], axis=1, errors='ignore')

    try:
        ids = db.return_existing_experiment_ids()
        if "experiment_id" in results_data.columns:
            results_data = results_data[
                ~results_data['experiment_id'].isin(ids)
            ]
        
    except Exception as e:
        pass

    results_data=results_data.iloc[start_row:end_row] 
    results_data['experiment_id'] = [
       str(stage)+"-"+ str(machine_id)+"-"+str(i)
        for i in range(start_row,end_row)
    ]
    
    return results_data
