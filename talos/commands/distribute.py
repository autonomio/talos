from ..scan.Scan import Scan
import json
import socket
import paramiko
import sys
import os
class DistributeScan(Scan):
    def __init__(self,
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
                 config_path='config.json',
                 file_path='script.py',
                 destination_path="/temp.py"
                 
                ):
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

        #distributed configurations
        self.config_path=config_path
        self.file_path=file_path
        self.destination_path=self.destination_path

        # input parameters section ends
    def load_config(self):
        config_path=self.config_path
        with open(config_path, 'r') as f:
          data = json.load(f)
        return data
    def load_params(self,param_file="params.json"):
        with open(os.path.join(os.getcwd(),param_file), 'r') as f:
          data = json.load(f)
        return data
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
        config=self.load_config()
        host = config['TALOS_IP_ADDRESS']
        port = config['TALOS_PORT']
        username = config['TALOS_USER']
        password = config['TALOS_PASSWORD']
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port, username, password)
        return client
        
    def ssh_run(self):
        client=self.ssh_connect()
        sftp = client.open_sftp()
        sftp.put(self.file_path, '{}'.format(self.destination_path))
        sftp.close()
        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdout = client.exec_command('python3 {}'.format(self.destination_path))[1]
        for line in stdout:
            # Process each line in the remote output
            print(line)

        client.close()
        sys.exit(0)
        
    def run_scan_new_params(self,p):
        super.__init__(  self.x,               
                         self.y,
                         p,
                         self.model,
                         self.experiment_name,
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
                         save_weights=self.save_weights,)
        
    def run(self):
        param_dicts=self.split_params(n_splits=2)#Change this to number of machine ids
        self.run_scan_new_params(param_dicts[0]) #run the scan in current machine using first split
        ### write the rest of the machine code here
            
                
