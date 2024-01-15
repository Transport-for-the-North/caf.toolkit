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
import sys
from pathlib import Path

# Third Party
from caf.toolkit.pandas_utils import sql_handling
import pandas as pd
import pyodbc

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
sys.path.append("..")
# # # CLASSES # # #


# # # FUNCTIONS # # #
@pytest.fixture(name="data_path", scope="session")
def fix_data_path():
    return Path(__file__).parent.resolve() / 'data'


@pytest.fixture(name="expected_simple", scope="session")
def fix_expected_simple(data_path):
    return pd.read_csv(data_path / "expected_simple.csv", index_col=[0, 1])


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


# @pytest.fixture(name="testing_db", scope="session")
# def fixture_testing_db():
#     connection = sqlite3.connect(":memory:")
#     cursor = connection.cursor()
#
#     # Create table1 with columns: id, name, age, and email
#     cursor.execute(
#         """
#         CREATE TABLE table1 (
#             id INTEGER PRIMARY KEY,
#             name TEXT,
#             age INTEGER,
#             email TEXT
#         )
#     """
#     )
#
#     # Create table2 with columns: id, table1_id, address, and phone
#     cursor.execute(
#         """
#         CREATE TABLE table2 (
#             id INTEGER PRIMARY KEY,
#             table1_id INTEGER,
#             address TEXT,
#             phone TEXT
#         )
#     """
#     )
#
#     # Create table3 with columns: id, table2_id, department, and position
#     cursor.execute(
#         """
#         CREATE TABLE table3 (
#             id INTEGER PRIMARY KEY,
#             table2_id INTEGER,
#             department TEXT,
#             position TEXT
#         )
#     """
#     )
#
#     # Insert sample data into table1
#     table1_data = [
#         (1, "John", 30, "john@example.com"),
#         (2, "Alice", 25, "alice@example.com"),
#         (3, "Bob", 35, "bob@example.com"),
#     ]
#     cursor.executemany(
#         "INSERT INTO table1 (id, name, age, email) VALUES (?, ?, ?, ?)", table1_data
#     )
#
#     # Insert sample data into table2
#     table2_data = [
#         (1, 1, "123 Main St", "123-456-7890"),
#         (2, 2, "456 Elm St", "987-654-3210"),
#         (3, 3, "789 Oak St", "555-123-4567"),
#     ]
#     cursor.executemany(
#         "INSERT INTO table2 (id, table1_id, address, phone) VALUES (?, ?, ?, ?)", table2_data
#     )
#
#     # Insert sample data into table3
#     table3_data = [
#         (1, 1, "HR", "Manager"),
#         (2, 2, "IT", "Developer"),
#         (3, 3, "Marketing", "Analyst"),
#     ]
#     cursor.executemany(
#         "INSERT INTO table3 (id, table2_id, department, position) VALUES (?, ?, ?, ?)",
#         table3_data,
#     )
#
#     # Commit the changes and close the connection
#     connection.commit()
#     # connection.close()
#
#     return connection


@pytest.fixture(name="tables", scope="session")
def fixture_tables():
    cols_1 = [
        sql_handling.Column(column_name="2021", groupby_fn="SUM"),
        sql_handling.Column(column_name="Mode"),
    ]
    cols_2 = [
        sql_handling.Column(column_name="Authority"),
    ]
    return [
        sql_handling.TableColumns(table_name="TripEndDataByDirection", columns=cols_1),
        sql_handling.TableColumns(table_name="Zones", columns=cols_2),
    ]


@pytest.fixture(name="group", scope="session")
def fixture_group():
    zones_group = [sql_handling.Column(column_name="Authority")]
    te_group = [sql_handling.Column(column_name="Mode")]
    return [
        sql_handling.TableColumns(table_name="Zones", columns=zones_group),
        sql_handling.TableColumns(table_name="TripEndDataByDirection", columns=te_group),
    ]


@pytest.fixture(name="joins", scope="session")
def fixture_joins():
    return [
        sql_handling.JoinInfo(
            left_table="TripEndDataByDirection",
            right_table="Zones",
            left_column="ZoneID",
            right_column="ZoneID",
        )
    ]


@pytest.fixture(name="query_test", scope="session")
def fixture_build_query_class(data_path, tables, joins, group):
    conf = sql_handling.MainSqlConf(
        file=data_path / "testing_ntem_db.mdb", tables=tables, joins=joins, groups=group
    )
    return sql_handling.QueryBuilder(conf)


@pytest.fixture(name="expected_out", scope="session")
def fixture_expected_out():
    return pd.DataFrame({"age": {1: 30, 2: 25, 3: 35}, "address": {1: 1, 2: 1, 3: 1}})


@pytest.fixture(name="where", scope="session")
def fixture_where():
    return [sql_handling.WhereInfo(table="table1", column="age", operator=">", match=25)]


@pytest.fixture(name="where_table", scope="session")
def fixture_where_table():
    return [
        sql_handling.TableColumns(
            table_name="table1",
            columns=[
                sql_handling.Column(column_name="age"),
                sql_handling.Column(column_name="email"),
            ],
        )
    ]


@pytest.fixture(name="expected_where", scope="session")
def fixture_expected_where():
    return pd.DataFrame({"age": {1: 30, 3: 35}, "address": {1: 1, 3: 1}})


@pytest.fixture(name="where_conf", scope="session")
def fixture_where_conf(where, where_table, testing_db):
    return sql_handling.MainSqlConf(file=testing_db, tables=where_table, wheres=where)


class TestBroad:
    def test_connection(self, query_test, expected_simple):
        read = query_test.load_db()
        assert expected_simple.astype("int").equals(read.astype("int"))

    # def test_where(self, where_conf, expected_where):
    #     assert sql_handling.QueryBuilder(where_conf).load_db().equals(expected_where)
