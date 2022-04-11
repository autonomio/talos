import time
import pandas as pd
from .distribute_utils import read_config, write_config, add_experiment_id, fetch_latest_file


def update_db(self, update_db_n_seconds, current_machine_id, remove_duplicates=True):
    '''

    Parameters
    ----------
    update_db_n_seconds | int |Time interval required to update the db
    remove_duplicates | bool |The default is True.

    Returns
    -------
    db| Database object | Database object with engine 

    '''

    # update the database every n seconds

    def __start_upload(config, results_data):
        from ..database.database import Database

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

                results_data = fetch_latest_file(self)

                if len(results_data) == 0 or len(results_data) == len(new_data):

                    start_time = new_time
                    time.sleep(update_db_n_seconds)
                    continue

                temp = results_data
                results_data = results_data[~results_data.isin(new_data)].dropna()
                new_data = temp

                if len(results_data) > 0:
                    results_data = add_experiment_id(
                        self, results_data, current_machine_id
                    )
                    __start_upload(config, results_data)

                new_config = read_config(self)

                if 'finished_scan_run' in new_config.keys():

                    results_data = fetch_latest_file(self)
                    results_data = results_data[
                        ~results_data.isin(new_data)
                    ].dropna()
                    results_data = add_experiment_id(
                        self, results_data, current_machine_id
                    )
                    __start_upload(config, results_data)
                    write_config(self, new_config)

                    print('Scan Run Finished in machine id ' + current_machine_id)

                    break

                else:

                    start_time = new_time
                    time.sleep(update_db_n_seconds)

            else:
                print('Database credentials not given.')

    del new_config['finished_scan_run']
    write_config(self, new_config)
