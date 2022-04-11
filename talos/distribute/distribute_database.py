import time
import pandas as pd
from .distribute_utils import read_config, write_config, add_experiment_id, fetch_latest_file

def get_db_object(self):
    config = self.config_data
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
    return db
    
def update_db(self, update_db_n_seconds, current_machine_id,stage):
    '''

    Parameters
    ----------
    update_db_n_seconds | int |Time interval required to update the db

    Returns
    -------
    db| Database object | Database object with engine 

    '''

    # update the database every n seconds
    db=get_db_object(self)
    config = self.config_data
    def __start_upload(results_data):
        if len(results_data) > 0:
            db.write_to_db(results_data)
        return db

    start_time = int(self.save_timestamp)

    start_row=0
    end_row=0
    
    while True:

        new_time = int(time.strftime('%D%H%M%S').replace('/', ''))

        if new_time - start_time >= update_db_n_seconds:

            if 'database' in config.keys():

                results_data = fetch_latest_file(self)
                
                if len(results_data) == 0:

                    start_time = new_time
                    time.sleep(update_db_n_seconds)
                    continue
                

                if len(results_data) > 0:
                    start_row=end_row
                    end_row=len(results_data)
  
                    if start_row!=end_row and end_row>start_row:
                        
                        results_data = add_experiment_id(
                            self, results_data, current_machine_id,start_row,end_row,db,stage
                        )
    
                        __start_upload( results_data)
                    
                if int(current_machine_id)==0:
                    remote=False
                else:
                    remote=True
                    
                new_config = read_config(self,remote)

                if 'finished_scan_run' in new_config.keys():

                    results_data = fetch_latest_file(self)
                    
                    start_row=end_row
                    end_row=len(results_data)

                    if start_row!=end_row and end_row>start_row:
                    
                        results_data = add_experiment_id(
                            self, results_data, current_machine_id,start_row,end_row,db,stage
                        )
                       
                        __start_upload(results_data)
                        write_config(self, new_config)
                        print('Scan Run Finished in machine id ' + current_machine_id)

                    exit()

                else:

                    start_time = new_time
                    time.sleep(update_db_n_seconds)

            else:
                print('Database credentials not given.')

    
