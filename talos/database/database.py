from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database, drop_database
from sqlalchemy.exc import DatabaseError
from sqlalchemy.schema import DropTable


class Database:
    def __init__(
        self,
        username=None,
        password=None,
        host=None,
        port=None,
        db_type='sqlite',
        database_name='EXPERIMENT_LOG',
        table_name='experiment_log',
        encoding='LATIN1',
    ):
        '''

        Parameters
        ----------
        username | str | The default is None.
        password | str | The default is None.
        host | str , optional| The default is None.
        port | str , optional| The default is None.
        db_type | str | The default is 'sqlite'.
        database_name | str | The default is 'EXPERIMENT_LOG'.
        table_name | str | The default is 'experiment_log'.

        Returns
        -------
        None.

        '''
        self.db_type = db_type
        self.database_name = database_name
        self.table_name = table_name
        self.encoding = encoding
        DB_URL = ''
        if db_type == 'sqlite':
            DB_URL = 'sqlite:///' + database_name + '.db'
        elif db_type == 'mysql':
            if port is None:
                port = 3306
            DB_URL = (
                'mysql+pymysql://'
                + username
                + ':'
                + password
                + '@'
                + host
                + ':'
                + str(port)
                + '/'
                + database_name
            )
        elif db_type == 'postgres':
            if port is None:
                port = 5432
            DB_URL = (
                'postgresql://'
                + username
                + ':'
                + password
                + '@'
                + host
                + ':'
                + str(port)
                + '/'
                + database_name
            )
        self.DB_URL = DB_URL

    def create_db(self):
        '''
        Create database if it doesn't exists.
        '''
        engine = create_engine(self.DB_URL, echo=False, isolation_level='AUTOCOMMIT')

        if not database_exists(engine.url):

            new_engine = create_engine(
                self.DB_URL.replace(self.database_name, ''),
                echo=False,
                isolation_level='AUTOCOMMIT',
            )
            conn = new_engine.connect()

            try:
                conn.execute(
                    '''
                    CREATE DATABASE {} ENCODING '{}'
                    '''.format(
                        self.database_name, self.encoding
                    )
                )

            except Exception as e:
                pass

        return engine

    def drop_db(self):
        '''
        Drop the database.
        '''
        drop_database(self.DB_URL)

    def delete_table(self, table_name):
        '''
        Delete the table.
        '''
        DropTable(table_name)

    def write_to_db(self, data_frame):
        '''


        Parameters
        ----------
        data_frame | `DataFrame` | DataFrame object consisting of tabular data.

        Returns
        -------
        None.

        '''
        engine = self.create_db()
        data_frame.to_sql(self.table_name, con=engine, if_exists='append', index=False)

    def query_table(self, query):
        '''


        Parameters
        ----------
        query | `str`| Database query for the respective sql engine

        Returns
        -------
        res | `list` of `tuples` | Query output from the database

        '''
        engine = self.create_db()
        res = engine.execute(query).fetchall()
        return res

    def show_table_content(self):
        '''


            Returns
            -------
        res |`list` of `tuples` | Query output from the database

        '''
        res = self.query_table('SELECT * FROM {}'.format(self.table_name))
        return res

    def return_table_df(self):
        '''


        Returns
        -------
        data | Pandas DataFrame object | returns the database as a dataframe

        '''
        import pandas as pd

        table = self.show_table_content()
        data = pd.DataFrame(table)
        return data

    def return_existing_experiment_ids(self):
        '''

        Returns
        -------
        ids | Pandas Series object | returns the experiment id of the table

        '''
        table = self.return_table_df()
        ids = table.iloc[:, -1]
        return ids
