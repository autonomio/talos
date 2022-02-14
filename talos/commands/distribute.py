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
                 config_path,
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
                 save_weights=True):
        self.x = x
        self.y = y
        self.params = params
        self.model = model
        self.experiment_name = experiment_name
        self.config_path=config_path
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
        # input parameters section ends
    def load_config(self):
        config_path=self.config_path
        with open(config_path, 'r') as f:
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
    def get_ip():
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
    def ssh_run(self):
        client=self.ssh_connect()
        sftp = client.open_sftp()
        path = os.getcwd()
        try:
            os.mkdir(path + '/' + self.experiment_name)
        except FileExistsError:
            pass
        sftp.put(__file__, path+'/{}/temp.py'.format(self.experiment_name))
        sftp.close()
        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdout = client.exec_command('python3 {}/{}/temp.py'.format(path,self.experiment_name))[1]
        for line in stdout:
            # Process each line in the remote output
            print(line)

        client.close()
        sys.exit(0)
