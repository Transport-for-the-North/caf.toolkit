"""
Created on: 02/08/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import pytest
import sqlite3
from unittest.mock import patch
# Third Party
from caf.toolkit.pandas_utils import sql_handling
import pandas as pd
import pyodbc
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
# @pytest.fixture
# def mock_database_connection():
#     with patch('pyodbc.connect') as mock_connect, patch('pyodbc.Cursor') as mock_cursor:
#         yield mock_connect, mock_cursor


@pytest.fixture(name="main_dir", scope="session")
def fixture_main_dir(tmp_path_factory):
    """
    Temporary path for I/O.

    Parameters
    ----------
    tmp_path_factory

    Returns
    -------
    Path: file path used for all saving and loading of files within the tests
    """
    path = tmp_path_factory.mktemp("main")
    return path

@pytest.fixture(name="testing_db", scope="session")
def fixture_testing_db(main_dir):
    mdb_file = main_dir / 'testing_db.mdb'
    # Define a System DSN (Data Source Name) to create the MDB file
    dsn_name = 'NewAccessDSN'
    mdb_driver = '{Microsoft Access Driver (*.mdb)}'

    # Create the system DSN
    pyodbc.win_create_mdb(dsn_name, mdb_file, mdb_driver)

    connection = pyodbc.connect(f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_file}')
    cursor = connection.cursor()

    # Create table1 with columns: id, name, age, and email
    cursor.execute("""
        CREATE TABLE table1 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            email TEXT
        )
    """)

    # Create table2 with columns: id, table1_id, address, and phone
    cursor.execute("""
        CREATE TABLE table2 (
            id INTEGER PRIMARY KEY,
            table1_id INTEGER,
            address TEXT,
            phone TEXT
        )
    """)

    # Create table3 with columns: id, table2_id, department, and position
    cursor.execute("""
        CREATE TABLE table3 (
            id INTEGER PRIMARY KEY,
            table2_id INTEGER,
            department TEXT,
            position TEXT
        )
    """)

    # Insert sample data into table1
    table1_data = [
        (1, "John", 30, "john@example.com"),
        (2, "Alice", 25, "alice@example.com"),
        (3, "Bob", 35, "bob@example.com")
    ]
    cursor.executemany("INSERT INTO table1 (id, name, age, email) VALUES (?, ?, ?, ?)",
                       table1_data)

    # Insert sample data into table2
    table2_data = [
        (1, 1, "123 Main St", "123-456-7890"),
        (2, 2, "456 Elm St", "987-654-3210"),
        (3, 3, "789 Oak St", "555-123-4567")
    ]
    cursor.executemany(
        "INSERT INTO table2 (id, table1_id, address, phone) VALUES (?, ?, ?, ?)", table2_data)

    # Insert sample data into table3
    table3_data = [
        (1, 1, "HR", "Manager"),
        (2, 2, "IT", "Developer"),
        (3, 3, "Marketing", "Analyst")
    ]
    cursor.executemany(
        "INSERT INTO table3 (id, table2_id, department, position) VALUES (?, ?, ?, ?)",
        table3_data)

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

    return mdb_file


@pytest.fixture(name="tables", scope="session")
def fixture_tables():
    cols_1 = [sql_handling.Column(column_name="id"),
              sql_handling.Column(column_name='age', groupby_fn='SUM')]
    cols_2 = [sql_handling.Column(column_name="table1_id"),
              sql_handling.Column(column_name='address', groupby_fn='COUNT')]
    return [sql_handling.TableColumns(table_name="table1", columns=cols_1),
            sql_handling.TableColumns(table_name="table2", columns=cols_2)]

@pytest.fixture(name="group", scope="session")
def fixture_group():
    group_col = [sql_handling.Column(column_name="id")]
    return [sql_handling.TableColumns(table_name="table1", columns=group_col)]
@pytest.fixture(name="joins", scope="session")
def fixture_joins():
    return [sql_handling.JoinInfo(left_table="table1", right_table="table2",
                                 left_column="id", right_column="table1_id")]


@pytest.fixture(name="query_test", scope="session")
def fixture_build_query_class(testing_db, tables, joins, group):
    conf = sql_handling.MainSqlConf(file=testing_db, tables=tables, joins=joins, groups=group)
    return sql_handling.QueryBuilder(conf)


@pytest.fixture(name="expected_out", scope="session")
def fixture_expected_out():
    return pd.DataFrame({'age': {1: 30, 2: 25, 3: 35},
                         'table1_id': {1: 1, 2: 2, 3: 3},
                         'address': {1: '123 Main St', 2: '456 Elm St', 3: '789 Oak St'}})


class TestBroad:
    def test_connection(self, query_test, expected_out):
        assert query_test.load_db().equals(expected_out)
