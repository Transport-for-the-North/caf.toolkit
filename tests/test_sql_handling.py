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
from pydantic_core import _pydantic_core

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
    return Path(__file__).parent.resolve() / "data"


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

@pytest.fixture(name="tables_fail", scope="session")
def fixture_tables_fail():
    cols_1 = [
        sql_handling.Column(column_name="2021", groupby_fn="SUM"),
        sql_handling.Column(column_name="Mode"),
    ]
    cols_2 = [
        sql_handling.Column(column_name="Authority"),
    ]
    return [
        sql_handling.TableColumns(table_name="_", columns=cols_1),
        sql_handling.TableColumns(table_name="_", columns=cols_2),
    ]


@pytest.fixture(name="group", scope="session")
def fixture_group():
    zones_group = [sql_handling.Column(column_name="Authority")]
    te_group = [sql_handling.Column(column_name="Mode")]
    return [
        sql_handling.TableColumns(table_name="Zones", columns=zones_group),
        sql_handling.TableColumns(table_name="TripEndDataByDirection", columns=te_group),
    ]

@pytest.fixture(name="group_fail", scope="session")
def fixture_group_fail():
    zones_group = [sql_handling.Column(column_name="Authority")]
    te_group = [sql_handling.Column(column_name="Mode")]
    return [
        sql_handling.TableColumns(table_name="_", columns=zones_group),
        sql_handling.TableColumns(table_name="_", columns=te_group),
    ]

@pytest.fixture(name="fail_join", scope="session")
def fixture_fail_join():
    return [
        sql_handling.JoinInfo(
            left_table="TripEndDataByDirection",
            right_table="Zones",
            left_column="ZoneID",
            right_column="_",
        )
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

@pytest.fixture(name="fail_join_test", scope="session")
def fixture_build_fail_join_class(data_path, tables, fail_join, group):
    conf = sql_handling.MainSqlConf(
        file=data_path / "testing_ntem_db.mdb", tables=tables, joins=fail_join, groups=group
    )
    return sql_handling.QueryBuilder(conf)

@pytest.fixture(name="fail_tables_test", scope="session")
def fixture_build_fail_tables_class(data_path, tables_fail, joins, group):
    conf = sql_handling.MainSqlConf(
        file=data_path / "testing_ntem_db.mdb", tables=tables_fail, joins=joins, groups=group
    )
    return sql_handling.QueryBuilder(conf)

@pytest.fixture(name="expected_columns", scope="session")
def fixture_expected_columns():
    return ["Authority", "Mode", "2021"]

@pytest.mark.skip
class TestSqlHandling:
    @pytest.mark.skip
    def test_connection(self, query_test):
        try:
            query_test.load_db()
        except Exception as e:
            raise ConnectionError(f"connection to db failed - {e}")
        assert True

    @pytest.mark.skip
    def test_columns(self, query_test, expected_columns):
        read = query_test.load_db().reset_index()

        for col in expected_columns:
            if col in read.columns:
                continue
            else:
                raise ValueError(f"expected {col} in output got {read.columns}")

    @pytest.mark.skip
    def test_fail_join(self, fail_join_test):
        with pytest.raises(pyodbc.ProgrammingError) as e_info:
            fail_join_test.load_db()

    @pytest.mark.skip
    def test_fail_join(self, data_path, tables_fail, joins, group):
         with pytest.raises(ValueError) as e_info:
            conf = sql_handling.MainSqlConf(
                file= data_path / "testing_ntem_db.mdb", tables=tables_fail, joins=joins, groups=group
            )
            sql_handling.QueryBuilder(conf)


    def test_fail_group(self, data_path, tables, joins, group_fail):

        with pytest.raises(ValueError) as e_info:
            conf = sql_handling.MainSqlConf(
                file=data_path / "testing_ntem_db.mdb", tables=tables, joins=joins, groups=group_fail
            )
            sql_handling.QueryBuilder(conf)


    def test_fail_data(self, data_path, tables, joins, group):

        with pytest.raises(pyodbc.Error) as e_info:
            conf = sql_handling.MainSqlConf(
                file=data_path / "Isaac_Scott_puts_the_milk_in_first.mdb", tables=tables, joins=joins, groups=group
            )
            sql_handling.QueryBuilder(conf)

    