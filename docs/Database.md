# Database()
The utilities required to connect to a database , read and write the experiment results to tables, etc.

```python
from talos import Database
db = Database(username,password,host,port=port,db_type,database_name)
```

The `Database` module can handle different databases including sqlite, postgres,and mysql. 

## Database Arguments

Parameter | type | Description
--------- | ------- | -----------
`username` | str | username for the database
`password` | str | password for the database
`host` | str | host ip address for the database
`port` | str | port number to connect with the database

## Database Package Contents

The database package consists of:

- `create_db()`:  create a database
- `drop_db()`:drop the database
- `delete_table(table_name)`: delete a given table name.
- `write_to_db(data_frame)`: write a dataframe to database.
- `query_table(query)`: run a database query in SQL syntax.
- `show_table_content()`: show the contents of a given table.
