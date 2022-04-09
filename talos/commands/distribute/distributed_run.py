import time
import json
import paramiko
import os
import pandas as pd
import inspect
from ...scan.Scan import Scan

def create_param_space(self, n_splits=2):
    '''
    Parameters
    ----------
    n_splits | int | The default is 2.

    Returns
    -------
    param_grid| `list` of `dict` | Split parameter spaces for each machine to run.

    '''

    from ...parameters.DistributeParamSpace import DistributeParamSpace

    params = self.params
    param_keys = self.params.keys()

    random_method = self.random_method
    fraction_limit = self.fraction_limit
    round_limit = self.round_limit
    time_limit = self.time_limit
    boolean_limit = self.boolean_limit

    param_grid = DistributeParamSpace(
        params=params,
        param_keys=param_keys,
        random_method=random_method,
        fraction_limit=fraction_limit,
        round_limit=round_limit,
        time_limit=time_limit,
        boolean_limit=boolean_limit,
        machines=n_splits,
    )

    return param_grid

def return_current_machine_id( self,):
    ''' return machine id after checking the ip from config'''

    current_machine_id = 0
    if 'current_machine_id' in self.config_data.keys():
        current_machine_id = int(self.config_data['current_machine_id'])

    print('Current machine ID is ' + str(current_machine_id))
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
    sftp.put('./new_config.json', self.dest_dir + '/config.json')
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

    ''' Run the transmitted script remotely without args and show its output.
     SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)'''

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

def run_central_machine(self, n_splits, run_central_node):
    '''

    Parameters
    ----------
    params  | `dict` | hyperparameter options

    Returns
    -------
    None.

    '''

    '''runs the experiment in central machine'''
    run_scan_with_split_params(self,n_splits, run_central_node)

def fetch_latest_file(self):
    '''fetch the latest csv for an experiment'''
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
    ''' create hashmap for a dataframe and use it for experiment id'''
    results_data = results_data.drop(['experiment_id'], axis=1, errors='ignore')
    results_data['experiment_id'] = pd.util.hash_pandas_object(results_data)
    results_data['experiment_id'] = [
        str(i) + '_machine_id_' + str(machine_id)
        for i in results_data['experiment_id']
    ]
    return results_data

def update_db(self, update_db_n_seconds, remove_duplicates=True):
    '''

    Parameters
    ----------
    update_db_n_seconds | int |Time interval required to update the db
    remove_duplicates | bool |The default is True.

    Returns
    -------
    db| Database object | Database object with engine 

    '''

    '''update the database every n seconds'''

    def __start_upload(config, results_data):
        from ...database.database import Database

        print('Starting database upload.....')

        machine_config = config['machines']
        db_config = config['database']
        username = db_config['DB_USERNAME']
        password = db_config['DB_PASSWORD']

        host_machine_id = int(db_config['DB_HOST_MACHINE_ID'])

        for machine in machine_config:
            if int(machine['machine_id']) == host_machine_id:
                host = machine['TALOS_IP_ADDRESS']
                break

        port = db_config['DB_PORT']
        database_name = db_config['DATABASE_NAME']
        db_type = db_config['DB_TYPE']
        table_name = db_config['DB_TABLE_NAME']
        encoding = db_config['DB_ENCODING']

        db = Database(
            username,
            password,
            host,
            port,
            database_name=database_name,
            db_type=db_type,
            table_name=table_name,
            encoding=encoding,
        )
        if remove_duplicates:
            try:
                experiment_ids = db.return_existing_experiment_ids()
                results_data = results_data[
                    ~results_data['experiment_id'].isin(experiment_ids)
                ]
            except:
                pass

        if len(results_data) > 0:
            db.write_to_db(results_data)
        return db

    start_time = int(self.save_timestamp)
    new_data = pd.DataFrame({})
    config = self.config_data

    while True:

        new_time = int(time.strftime('%D%H%M%S').replace('/', ''))

        if new_time - start_time >= update_db_n_seconds:

            if 'database' in config.keys():

                print(
                    'Updating to db every ' + str(update_db_n_seconds) + ' seconds'
                )

                results_data = fetch_latest_file(self)

                if len(results_data) == 0 or len(results_data) == len(new_data):
                    print('Waiting for rounds to finish......')
                    start_time = new_time
                    time.sleep(update_db_n_seconds)
                    continue

                temp = results_data
                results_data = results_data[~results_data.isin(new_data)].dropna()
                new_data = temp

                if len(results_data) > 0:
                    current_machine_id = str(return_current_machine_id(self))
                    results_data = add_experiment_id(
                       self,results_data, current_machine_id
                    )
                    database_object = __start_upload(config, results_data)

                new_config = read_config(self)

                if 'finished_scan_run' in new_config.keys():

                    results_data = fetch_latest_file(self)
                    results_data = results_data[
                        ~results_data.isin(new_data)
                    ].dropna()
                    current_machine_id = str(return_current_machine_id(self))
                    results_data = add_experiment_id(
                       self,results_data, current_machine_id
                    )
                    database_object = __start_upload(config, results_data)
                    write_config(self,new_config)

                    print('Scan Run Finished in machine id ' + current_machine_id)

                    break

                else:

                    start_time = new_time
                    time.sleep(update_db_n_seconds)

            else:
                print('Database credentials not given.')

    del new_config['finished_scan_run']
    write_config(self,new_config)

def run_scan_with_split_params(self, machines=2, run_central_node=False):
    '''


    Parameters
    ----------
    machines | int |The default is 2.
    run_central_node | bool | The default is False.

    Returns
    -------
    None.

    '''
    '''runs Scan a machine after param split'''
    machine_id = return_current_machine_id(self)

    if not run_central_node:
        if machine_id != 0:
            machine_id = machine_id - 1

    split_params = create_param_space(self,n_splits=machines).param_spaces[
        machine_id
    ]

    scan_object = Scan(
        x=self.x,
        y=self.y,
        params=split_params,  # the split params used for Scan
        model=self.model,
        experiment_name=self.experiment_name,
        x_val=self.x_val,
        y_val=self.y_val,
        val_split=self.val_split,
        random_method=self.random_method,
        seed=self.seed,
        performance_target=self.performance_target,
        fraction_limit=self.fraction_limit,
        round_limit=self.round_limit,
        time_limit=self.time_limit,
        boolean_limit=self.boolean_limit,
        reduction_method=self.reduction_method,
        reduction_interval=self.reduction_interval,
        reduction_window=self.reduction_window,
        reduction_threshold=self.reduction_threshold,
        reduction_metric=self.reduction_metric,
        minimize_loss=self.minimize_loss,
        disable_progress_bar=self.disable_progress_bar,
        print_params=self.print_params,
        clear_session=self.clear_session,
        save_weights=self.save_weights,
    )

    new_config = read_config(self)
    new_config['finished_scan_run'] = True
    if machine_id == 0:
        new_config['current_machine_id'] = 0
    write_config(self,new_config)



def distributed_run(self):
    
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
    
    #run the Scan script in distributed machines
    
    
    import json
    import threading
    
    config = self.config_data
    
    if 'run_central_node' in config.keys():
        run_central_node=config['run_central_node']
    else:
        run_central_node=False
    
    update_db_n_seconds=5
    if 'DB_UPDATE_INTERVAL' in config['database'].keys():
        update_db_n_seconds = int(config['database']['DB_UPDATE_INTERVAL'])
    
    current_machine_id = return_current_machine_id(self)
    n_splits = len(config['machines'])
    
    if run_central_node:
        n_splits += 1
    
    if current_machine_id == 0:
    
        clients = ssh_connect(self)
    
        for machine_id, client in clients.items():
            new_config = config
            new_config['current_machine_id'] = machine_id
            with open('new_config.json', 'w') as outfile:
                json.dump(new_config, outfile)
            ssh_file_transfer(self,client, machine_id)
        threads = []
    
        if run_central_node:
            print('Running Scan in Central Node....')
            t = threading.Thread(
                target=run_central_machine,
                args=(self,n_splits, run_central_node),
            )
            t.start()
            threads.append(t)
    
            t = threading.Thread(
                target=update_db,
                args=([self,update_db_n_seconds]),
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
    
    # else:
    #     threads = []
    
    #     t = threading.Thread(
    #         target=update_db,
    #         args=([self,update_db_n_seconds]),
    #     )
    #     t.start()
    #     threads.append(t)
    
    #     t = threading.Thread(
    #         target=run_scan_with_split_params,
    #         args=(self,n_splits, run_central_node),
    #     )
    #     t.start()
    #     threads.append(t)
    
    #     for t in threads:
    #         t.join()
