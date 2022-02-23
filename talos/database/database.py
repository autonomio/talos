from sqlalchemy import create_engine, types
from sqlalchemy_utils import database_exists, create_database, drop_database
from sqlalchemy.exc import DatabaseError
from sqlalchemy.schema import DropTable
import ast
import pandas as pd



class Database:

    def __init__(self,username,password,host,port=None,db_type="postgres",database_name="EXPERIMENT_LOG",table_name="experiment_log"):
     

        self.db_type=db_type
        self.database_name=database_name
        self.table_name=table_name
        DB_URL=""
        if db_type == 'sqlite':
            DB_URL = 'sqlite://'
        elif db_type == 'mysql':
            if port is None:    port = 3306
            DB_URL = 'mysql+pymysql://' + username + ':' + password + '@' + host + ':' + str(port)+'/'+database_name
        elif db_type == 'postgres':
            if port is None:    port = 5432
            DB_URL = 'postgresql://' + username + ':' + password + '@' + host + ':' + str(port) + '/' + database_name
        self.DB_URL=DB_URL
    def create_db(self):
        """
        Create database if it doesn't exists.
        """
        try:
            engine = create_engine(self.DB_URL, echo=False)
            if not database_exists(engine.url):
                create_database(engine.url)
        except DatabaseError as e:
            import traceback
            traceback.print_exc()
            raise Exception("Check whether the server is running and the connection parameters are correct.")        
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception("Some error occured while connecting to the database.")
        return engine

    def drop_db(self):
        """
        Drop the database.
        """
        drop_database(self.DB_URL)

    def delete_table(self, table_name):
        """
        Delete the table.
        """
        DropTable(table_name)
        
    def write_to_db(self, data_frame):
        engine = self.create_db()
        data_frame.to_sql(self.table_name, con=engine, if_exists='append')
    def query_table(self,query):
        engine = self.create_db()
        res=engine.execute(query).fetchall()
        return res
    def show_table_content(self):
        res=self.query_table("SELECT * FROM {}".format(self.table_name))
        return res
        
        