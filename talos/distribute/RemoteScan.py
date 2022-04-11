from ..scan.Scan import Scan

import time
import json


class RemoteScan(Scan):
    def __init__(
        self,
        x,
        y,
        params,
        model,
        experiment_name,
        x_val=None,
        y_val=None,
        val_split=0.3,
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
        config='tmp/remote_config.json',
    ):
        '''


        Parameters
        ----------
        params | `dict` | Hyperparameters for distribution.
        config | str or dict | The default is 'tmp/remote_config.json'.

        Returns
        -------
        None.

        '''
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

        self.save_timestamp = time.strftime('%D%H%M%S').replace('/', '')

        if type(config) == str:
            with open(config, 'r') as f:
                self.config_data = json.load(f)

        elif type(config) == dict:
            self.config_data = config
            with open('tmp/remote_config.json', 'w') as outfile:
                json.dump(self.config_data, outfile, indent=2)

        else:
            TypeError('''Enter config path or config dict''')

        if 'finished_scan_run' in self.config_data.keys():
            del self.config_data['finished_scan_run']

        config = self.config_data
        
        with open("tmp/arguments_remote.json" ,"r") as f:
            arguments_dict=json.load(f)
            
        self.stage=arguments_dict["stage"]
        
        if 'run_central_node' in config.keys():
            run_central_node = config['run_central_node']
        else:
            run_central_node = False

        from .distribute_params import run_scan_with_split_params
        from .distribute_database import update_db
        from .distribute_utils import return_current_machine_id

        import threading

        n_splits = len(config['machines'])
        if run_central_node:
            n_splits += 1

        update_db_n_seconds = 5
        if 'DB_UPDATE_INTERVAL' in config['database'].keys():
            update_db_n_seconds = int(config['database']['DB_UPDATE_INTERVAL'])

        current_machine_id = str(return_current_machine_id(self))

        threads = []

        t = threading.Thread(
            target=update_db,
            args=([self, update_db_n_seconds, current_machine_id,self.stage]),
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=run_scan_with_split_params,
            args=(self, n_splits, run_central_node, current_machine_id),
        )
        t.start()
        threads.append(t)

        for t in threads:
            t.join()
