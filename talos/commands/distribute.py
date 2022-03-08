from ..scan.Scan import Scan

import json
import socket
import paramiko
import sys
import os
import threading
import datetime
import pandas as pd
class DistributeScan(Scan):
    def __init__(self,
                 params,
                 config='config.json',
                 file_path='script.py',
                 destination_path="./temp.py",
                 experiment_name="talos_experiment"
                 
                ):
        #distributed configurations
        self.params = params
        self.config=config
        self.file_path=file_path
        self.destination_path=destination_path
        self.experiment_name=experiment_name
        self.dest_dir=os.path.dirname(self.destination_path)+"/"+self.experiment_name
        self.save_timestamp = int(datetime.datetime.now().timestamp())

        # input parameters section ends
    def load_config(self):
        config=self.config
        if type(config)==str:
            with open(config, 'r') as f:
              data = json.load(f)
            return data["machines"]
        elif type(config)==dict:
            return config["machines"]
        else:
            TypeError("Please enter the config path or pass the config parameters as a dictionary")
   
    def create_param_space(self,n_splits=2):
        
        from ..parameters.ParamSpace import ParamSpace
        params=self.params
        param_keys=params.keys()
        param_grid= ParamSpace(params, param_keys)._param_space_creation()
        
        def __column(matrix, i):
            return [row[i] for row in matrix]
        
        new_params={k:[] for k in param_keys}
        for key_index,key in enumerate(param_keys):
            new_params[key]=__column(param_grid,key_index)
            
        def __split_params(n_splits=n_splits):
            d=new_params
            dicts=[{} for i in range(n_splits)]
            def _chunkify(lst,n):
              return [lst[i::n] for i in range(n)]
            for k,v in d.items():
                for i in range(n_splits):
                    dicts[i][k]=_chunkify(v, n_splits)[i]
            return dicts
        
        new_params=__split_params(n_splits)
        return new_params
        

    def ssh_connect(self):
        configs=self.load_config()
        clients=[]
        for config in configs:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            host = config['TALOS_IP_ADDRESS']
            port = config['TALOS_PORT']
            username = config['TALOS_USER']
            if "TALOS_PASSWORD" in config.keys():
                password = config['TALOS_PASSWORD']
                client.connect(host, port, username, password)
            elif "TALOS_KEY_FILENAME" in config.keys():
                client.connect(host, port, username, key_filename=config["TALOS_KEY_FILENAME"])
            clients.append(client)
        return clients
    def ssh_run(self,client,params,machine_id):
        sftp = client.open_sftp()
        sftp.put(self.file_path, '{}'.format(self.destination_path))
        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdin,stdout,stderr = client.exec_command('python3 {} "{}" {}'.format(self.destination_path,params,self.dest_dir))
        for line in stderr:
            # Process each line in the remote output
            print(line)
        for line in stdout:
            print(line)
            
        #fetch the latest csv
        localpath = self.experiment_name
        remotepath = self.dest_dir

        sftp.chdir(remotepath)
        for f in sorted(sftp.listdir_attr(), key=lambda k: k.st_mtime, reverse=True):
            sftp.get(f.filename,localpath+"/"+str(self.save_timestamp)+"_machine_"+str(machine_id)+".csv")
            break

        sftp.close()
        client.close()
        
    def run_local(self,params):
        os.system('python3 {} "{}" {} '.format(self.file_path,params,self.dest_dir))
               
    def merge_csvs(self):
        localpath=self.experiment_name
        filepaths=[os.path.join(localpath,i) for i in os.listdir(localpath) if i.startswith(str(self.save_timestamp))]
        dfs=[pd.read_csv(f) for f in filepaths]
        results=pd.concat(dfs)
        results.to_csv(os.path.join(localpath,str(self.save_timestamp)+"_results.csv"))
        return results
    
    def update_db(self,data_frame,config=None):
        from ..database.database import Database
        if config:
            username=config["DB_USERNAME"]
            password=config["DB_PASSWORD"]
            host=config["DB_HOST"]
            port=config["DB_PORT"]
            db=Database(username,password,host,port)
        else:
            db=Database()
        db.write_to_db(data_frame)
        return db
        
        
    def distributed_run(self,run_local=False,db_machine_id=0,db_config=None,show_results=False):
        """
        run the file in distributed systems. 
        Uses threading in the main machine to connect to multiple systems. 

        Parameters
        ----------
        run_local : TYPE, optional
            DESCRIPTION. The default is False.
        db_machine_id: int
            DESCRIPTION. The default is 0. Indicates the centralised store where
                         the data gets merged.

        Returns
        -------
        None.

        """
        clients=self.ssh_connect()
        n_splits=len(clients)
        threads=[]
        
        if run_local:
            n_splits+=1
            params_dict=self.create_param_space(n_splits=n_splits)
            params=params_dict[0]
            t = threading.Thread(target=self.run_local, args=(params,))
            t.start()
            threads.append(t)
            params_dict=params_dict[1:]
        else:
            params_dict=self.create_param_space(n_splits=n_splits)
            
        for machine_id,client in enumerate(clients):
            t = threading.Thread(target=self.ssh_run, args=(client,params_dict[machine_id],machine_id+1))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
        
        results=self.merge_csvs()
        db=self.update_db(results,db_config)
        if show_results:
            print(db.show_table_content())
        
    
            
            
            
                
