from ..scan.Scan import Scan

import time
import json
import paramiko
import os
import threading
import datetime
import pandas as pd


class DistributeScan(Scan):
    def __init__(
        self,
        x,
        y,
        params,
        model,
        experiment_name,
        x_val=None,
        y_val=None,
        val_split=.3,
        random_method='uniform_mersenne',
        seed=None,
        performance_target=None,
        fraction_limit=None,
        round_limit=None,
        time_limit=None,
        boolean_limit=None,
        reduction_method=None,
        reduction_interval=50,
        reduction_window=20,
        reduction_threshold=0.2,
        reduction_metric='val_acc',
        minimize_loss=False,
        disable_progress_bar=False,
        print_params=False,
        clear_session=True,
        save_weights=True,
        config="./config.json",
        file_path="./temp.py",
        destination_path="./newtemp.py",
        local_distribute_path="./talos_distribute.py",
        remote_distribute_filepath="./talos_distribute.py",
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
        self.x = x
        self.y = y
        self.params = params
        self.model = model
        self.experiment_name = experiment_name
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split

        # randomness
        self.random_method = random_method
        self.seed = seed

        # limiters
        self.performance_target = performance_target
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.boolean_limit = boolean_limit

        # optimization
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.reduction_threshold = reduction_threshold
        self.reduction_metric = reduction_metric
        self.minimize_loss = minimize_loss

        # display
        self.disable_progress_bar = disable_progress_bar
        self.print_params = print_params

        # performance
        self.clear_session = clear_session
        self.save_weights = save_weights
        
        # distributed configurations
        self.config = config
        self.file_path = file_path
        self.destination_path = destination_path
        self.remote_distribute_filepath = remote_distribute_filepath
        self.local_distribute_path = local_distribute_path

        self.dest_dir = (
            os.path.dirname(self.destination_path)
        )
        self.save_timestamp =  time.strftime('%D%H%M%S').replace('/', '')

        if type(config) == str:
            with open(config, "r") as f:
                self.config_data = json.load(f)

        elif type(config) == dict:
            self.config_data = config
            with open("config.json", "w") as outfile:
                json.dump(self.config_data, outfile)
                    

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

        from ..parameters.DistributeParamSpace import DistributeParamSpace

        params = self.params
        param_keys = self.params.keys()
        
        random_method=self.random_method
        fraction_limit=self.fraction_limit
        round_limit=self.round_limit
        time_limit=self.time_limit
        boolean_limit=self.boolean_limit
        
        param_grid = DistributeParamSpace(
            params=params,
            param_keys=param_keys,
            random_method=random_method,
            fraction_limit=fraction_limit,
            round_limit=round_limit,
            time_limit=time_limit,
            boolean_limit=boolean_limit,
            machines=n_splits
            )
        
        return param_grid
    
    def return_current_machine_id(self):# return machine id after checking the ip from config

        current_machine_id=0
        if "current_machine_id" in self.config_data.keys():
            current_machine_id=int(self.config_data["current_machine_id"])
        
        print("Current machine ID is "+str(current_machine_id))
        return current_machine_id
    
    def return_central_machine_id(self):
        central_id=0
        config_data=self.config_data
        if "database" in config_data.keys():
            central_id=int(config_data["database"]["DB_HOST_MACHINE_ID"])
        return central_id   
        
    def read_config(self):
        with open("config.json", "r") as f:
            config_data = json.load(f)
        return config_data
        
    def ssh_connect(self):
        """


        Returns
        -------
        clients : TYPE : `list`
            DESCRIPTION. List of client objects of machines after connection.

        """
        configs = self.config_data["machines"]
        clients = {}
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
                    host, port, username, key_filename=config["TALOS_KEY_FILENAME"]
                )

            clients[config["machine_id"]]=client
        return clients
    
    def ssh_file_transfer(self,client,machine_id):
        sftp = client.open_sftp()
        sftp.put(self.file_path, self.destination_path)
        sftp.put("./new_config.json",self.dest_dir+"/config.json")
        sftp.close()

    def ssh_run(self, client,machine_id):
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
        # current_machine_id=int(self.return_current_machine_id())
        # central_machine_id=int(self.return_central_machine_id())

            # Run the transmitted script remotely without args and show its output.
            # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        
        stdin, stdout, stderr = client.exec_command(
            'python3 {}'.format(self.destination_path)
        )
        if stderr:
            for line in stderr:
                try:
                    # Process each error  line in the remote output
                    print(line)
                except:
                    print("Can't Output error")

        for line in stdout:
            try:
                # Process each  line in the remote output
                print(line)
            except:
                print("Can't Output error")
     


    def run_local(self, n_splits):
        """


        Parameters
        ----------
        params : TYPE: `dict`
            DESCRIPTION. hyperparameter options

        Returns
        -------
        None.

        """
        self.run_distributed_scan(machines=n_splits,machine_id=0)
    
    def fetch_latest_file(self):
   
        experiment_name=self.experiment_name
        save_timestamp=self.save_timestamp
        

        filelist=[os.path.join(experiment_name,i) for i in os.listdir(experiment_name) if i.endswith(".csv") and int(i.replace(".csv",""))>=int(save_timestamp)]

        if filelist:
                
            latest_filepath=max(filelist, key=os.path.getmtime)
            
            
            try:
                results_data=pd.read_csv(latest_filepath)
            except Exception as e:
                print("File empty..waiting...")
                return []
            
            return results_data
        
        else:
            return []
            
    
    def update_db(self,update_db_n_seconds=5):
        
        start_time=int(self.save_timestamp)
        
        
        while True:
            
            new_time=int(time.strftime('%D%H%M%S').replace('/', ''))
            
            
            
            if new_time-start_time>=update_db_n_seconds:

                
                if "database" in self.config_data.keys():
                    
                    results_data=self.fetch_latest_file()
                    
                    if  len(results_data)==0:
                        print("Waiting for experiment to finish.......")
                        start_time=new_time
                        time.sleep(update_db_n_seconds)
                        continue
                    
                    print("Starting database upload.....")
                    from ..database.database import Database
                    
                    config=self.config_data
                    machine_config = config["machines"]
                    db_config = config["database"]
                    username = db_config["DB_USERNAME"]
                    password = db_config["DB_PASSWORD"]
        
                    host_machine_id = (
                        int(db_config["DB_HOST_MACHINE_ID"]) 
                    ) 
                    
                    for machine in machine_config:
                        if int(machine["machine_id"])==host_machine_id:
                            host = machine["TALOS_IP_ADDRESS"]
                            break
        
                    port = db_config["DB_PORT"]
                    database_name = db_config["DATABASE_NAME"]
                    db_type = db_config["DB_TYPE"]
                    table_name = db_config["DB_TABLE_NAME"]
                    encoding = db_config["DB_ENCODING"]
                    
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
                    
                    db.write_to_db(results_data)
                    
                    self.database_object=db
                    
                    if "finished_scan_run" in self.read_config().keys():
                        print("Scan Run Finished")
                        break
                    
                    else:
                        print("Updating to db every "+str(update_db_n_seconds)+" seconds")
                        start_time=new_time
                        time.sleep(update_db_n_seconds)
            
                else:
                    print("Database credentials not given.")
        
        

    def run_distributed_scan(self,machines=2,machine_id=None,show_results=False):
        if machine_id!=0: #machine id for non central nodes since param split starts from 0
            machine_id=machine_id-1 
            
        split_params=self.create_param_space(n_splits=machines).param_spaces[machine_id]

        scan_object=Scan(
            x = self.x,
            y = self.y,
            params = split_params, #the split params used for Scan
            model = self.model,
            experiment_name = self.experiment_name,
            x_val = self.x_val,
            y_val = self.y_val,
            val_split = self.val_split,
            random_method = self.random_method,
            seed = self.seed,

            performance_target = self.performance_target,
            fraction_limit = self.fraction_limit,
            round_limit = self.round_limit,
            time_limit = self.time_limit,
            boolean_limit = self.boolean_limit,


            reduction_method = self.reduction_method,
            reduction_interval = self.reduction_interval,
            reduction_window = self.reduction_window,
            reduction_threshold = self.reduction_threshold,
            reduction_metric = self.reduction_metric,
            minimize_loss = self.minimize_loss,
            disable_progress_bar = self.disable_progress_bar,
            print_params = self.print_params,

            clear_session = self.clear_session,
            save_weights = self.save_weights,
            )
        
        new_config={}
        new_config["finished_scan_run"]=True
        with open("config.json", "w") as outfile:
            json.dump(new_config, outfile)
            
        if show_results:
            if hasattr(self,'database_object'):
                db=self.database_object
                table_content=db.show_table_content()
                
                print(table_content)
                print(len(table_content))
     
        

    def distributed_run(self,run_central_node=False, show_results=False,update_db_n_seconds=5):
        """


        Parameters
        ----------
        run_local : TYPE: `bool`, optional
            DESCRIPTION. The default is False.
        db_machine_id:TYPE : `int`
            DESCRIPTION. The default is 0. Indicates the centralised store
                          where the data gets merged.
        show_results : TYPE: `bool`, optional
            DESCRIPTION. The default is False. Shows results from database.

        Returns
        -------
        None.

        """

        config = self.config_data
        machine_id=self.return_current_machine_id()
        n_splits = len(config["machines"]) 
        if run_central_node:
            n_splits += 1
            
        if machine_id==0:
            
            clients = self.ssh_connect()
            for machine_id, client in clients.items():
                new_config=config
                new_config["current_machine_id"]=machine_id
                with open("new_config.json", "w") as outfile:
                    json.dump(new_config, outfile)
                self.ssh_file_transfer(client, machine_id)
            threads = []
            
            if run_central_node:
                print("Running Scan in Central Node....")
                t = threading.Thread(target=self.run_local, args=(n_splits,))
                t.start()
                threads.append(t)
                
                t = threading.Thread(
                    target=self.update_db,
                    args=([update_db_n_seconds]),
                )
                t.start()
                threads.append(t)
    
            for machine_id, client in clients.items():
                t = threading.Thread(
                    target=self.ssh_run,
                    args=(client,machine_id,),
                )
                t.start()
                threads.append(t)
            
                
            for t in threads:
                t.join()
                
        else:
            threads=[]
            
            t = threading.Thread(
                target=self.update_db,
                args=([update_db_n_seconds]),
            )
            t.start()
            threads.append(t)
            
            t = threading.Thread(
                target=self.run_distributed_scan,
                args=(n_splits,machine_id,show_results),
            )
            t.start()
            threads.append(t)
            
            for t in threads:
                t.join()
            # self.run_distributed_scan(machines=n_splits,machine_id=machine_id,show_results=show_results)
            
        
       

        


