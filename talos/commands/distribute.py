from ..scan.Scan import Scan
import json
import socket
import paramiko
import sys
import os
import threading
class DistributeScan(Scan):
    def __init__(self,
                 params,
                 config_path='config.json',
                 file_path='script.py',
                 destination_path="./temp.py"
                 
                ):
        #distributed configurations
        self.params = params
        self.config_path=config_path
        self.file_path=file_path
        self.destination_path=destination_path

        # input parameters section ends
    def load_config(self):
        config_path=self.config_path
        with open(config_path, 'r') as f:
          data = json.load(f)
        return data["machines"]
    def split_params(self,n_splits=2):
        d=self.params
        dicts=[{} for i in range(n_splits)]
        def _chunkify(lst,n):
          return [lst[i::n] for i in range(n)]
        for k,v in d.items():
            for i in range(n_splits):
                dicts[i][k]=_chunkify(v, n_splits)[i]
        return dicts

    def ssh_connect(self):
        configs=self.load_config()
        clients=[]
        for config in configs:
            host = config['TALOS_IP_ADDRESS']
            port = config['TALOS_PORT']
            username = config['TALOS_USER']
            password = config['TALOS_PASSWORD']
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, port, username, password)
            clients.append(client)
        return clients
    def ssh_run(self,client,params):
        sftp = client.open_sftp()
        sftp.put(self.file_path, '{}'.format(self.destination_path))
        sftp.close()
        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdin,stdout,stderr = client.exec_command('python3 {} "{}" '.format(self.destination_path,params))

        for line in stderr:
            # Process each line in the remote output
            print(line)
        for line in stdout:
            print(line)
        client.close()
    def run_local(self,params):
        os.system('python3 {} "{}" '.format(self.file_path,params))

    def distributed_run(self,run_local=False):
        clients=self.ssh_connect()
        n_splits=len(clients)
        threads=[]
        
        if run_local:
            n_splits+=1
            params_list=self.split_params(n_splits=n_splits)
            params=params_list[0]
            t = threading.Thread(target=self.run_local, args=(params,))
            t.start()
            threads.append(t)
            params_list=params_list[1:]
        else:
            params_list=self.split_params(n_splits=n_splits)
            
        for machine_id,client in enumerate(clients):
            t = threading.Thread(target=self.ssh_run, args=(client,params_list[machine_id],))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
    
            
            
            
                
