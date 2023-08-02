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
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
@pytest.fixture
def mock_database_connection():
    with patch('pyodbc.connect') as mock_connect, patch('pyodbc.Cursor') as mock_cursor:
        yield mock_connect, mock_cursor

@pytest.fixture(name="testing_db", scope="module")
def fixture_testing_db():
    # Connect to an in-memory SQLite database
    connection = sqlite3.connect(":memory:")
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

    return connection


@pytest.fixture(name="tables", scope="session")
def fixture_tables():
    cols_1 = [sql_handling.Column(column_name="id"),
              sql_handling.Column(column_name='age', groupby_fn='SUM')]
    cols_2 = [sql_handling.Column(column_name="table_1_id"),
              sql_handling.Column(column_name='department', groupby_fn='COUNT')]
    return [sql_handling.TableColumns(table_name="table_1", columns=cols_1),
            sql_handling.TableColumns(table_name="table_2", columns=cols_2)]

@pytest.fixture(name="group", scope="session")
def fixture_group():
    group_col = [sql_handling.Column(column_name="id")]
    return [sql_handling.TableColumns(table_name="table_1", columns=group_col)]
@pytest.fixutre(name="joins", scope="session")
def fixture_joins():
    return sql_handling.JoinInfo(left_table="table_1", right_table="table_2",
                                 left_column="id", right_column="table_1_id")

@pytest.fixture(name="query_test", scope="session")
def fixture_build_query_class(testing_db, tables, joins, group):
    return sql_handling.QueryBuilder(testing_db, tables, joins, group=group)
class TestBroad:
    def test_connection(self, query_test):
        assert query_test == 5

# You can also define other fixtures or setup/teardown functions as needed.
