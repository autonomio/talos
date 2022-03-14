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
        db_type="sqlite",
        database_name="EXPERIMENT_LOG",
        table_name="experiment_log",
        encoding="LATIN1"
    ):
        """

        Parameters
        ----------
        username : TYPE, optional
            DESCRIPTION. The default is None.
        password : TYPE, optional
            DESCRIPTION. The default is None.
        host : TYPE, optional
            DESCRIPTION. The default is None.
        port : TYPE, optional
            DESCRIPTION. The default is None.
        db_type : TYPE, optional
            DESCRIPTION. The default is "sqlite".
        database_name : TYPE, optional
            DESCRIPTION. The default is "EXPERIMENT_LOG".
        table_name : TYPE, optional
            DESCRIPTION. The default is "experiment_log".

        Returns
        -------
        None.

        """
        self.db_type = db_type
        self.database_name = database_name
        self.table_name = table_name
        self.encoding=encoding
        DB_URL = ""
        if db_type == "sqlite":
            DB_URL = "sqlite:///" + database_name + ".db"
        elif db_type == "mysql":
            if port is None:
                port = 3306
            DB_URL = (
                "mysql+pymysql://"
                + username
                + ":"
                + password
                + "@"
                + host
                + ":"
                + str(port)
                + "/"
                + database_name
            )
        elif db_type == "postgres":
            if port is None:
                port = 5432
            DB_URL = (
                "postgresql://"
                + username
                + ":"
                + password
                + "@"
                + host
                + ":"
                + str(port)
                + "/"
                + database_name
            )
        self.DB_URL = DB_URL

    def create_db(self):
        """
        Create database if it doesn't exists.
        """
        try:
            
            engine = create_engine(self.DB_URL, echo=False,
                                  isolation_level="AUTOCOMMIT" 
                                  )
            
            if not database_exists(engine.url):
                
                new_engine = create_engine(self.DB_URL.replace(self.database_name,""), echo=False,
                                      isolation_level="AUTOCOMMIT" 
                                      )
                conn=new_engine.connect()
                conn.execute("""
                             CREATE DATABASE {} ENCODING '{}' 
                             """.format(self.database_name,
                                        self.encoding))
            
        except DatabaseError as e:
            import traceback

            traceback.print_exc()
            raise Exception(
                "Check whether the server is running and the connection parameters are correct."
            )
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
        """


        Parameters
        ----------
        data_frame : TYPE: `DataFrame`
            DESCRIPTION. DataFrame object consisting of tabular data.

        Returns
        -------
        None.

        """
        engine = self.create_db()
        data_frame.to_sql(self.table_name, con=engine, if_exists="append",index=False)

    def query_table(self, query):
        """


        Parameters
        ----------
        query : TYPE: `str`
            DESCRIPTION. Database query for the respective sql engine

        Returns
        -------
        res : TYPE: `list` of `tuples`
            DESCRIPTION. Query output from the database

        """
        engine = self.create_db()
        res = engine.execute(query).fetchall()
        return res

    def show_table_content(self):
        """


            Returns
            -------
        res : TYPE: `list` of `tuples`
            DESCRIPTION. Query output from the database

        """
        res = self.query_table("SELECT * FROM {}".format(self.table_name))
        return res