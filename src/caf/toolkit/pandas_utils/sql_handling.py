from caf.toolkit import BaseConfig
from pathlib import Path
from typing import Union
import pyodbc
import pandas as pd
from pydantic import validator

STRINGTYPENAMES = ("VARCHAR", "TEXT", "MEMO", "DATETIME", "YESNO", "CHARACTER")

aggregate_functions = (
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "FIRST",
    "LAST",
    "STDDEV",
    "STDEV_POP",
    "VARIANCE",
    "VAR_POP",
    "MEDIAN",
    "GROUP_CONCAT",
    "STRING_AGG",
    "ANY",
    "SOME",
    "ALL",
    "EVERY",
    "RANK",
    "DENSE_RANK",
    "PERCENT_RANK",
    "CUME_DIST",
    "ROW_NUMBER",
)


class Column(BaseConfig):
    """
    Info about a column of a table, for use in TableColumns.
    """

    column_name: str
    groupby_fn: str = None

    @validator("groupby_fn")
    def valid_aggregate(cls, val):
        if val.upper() not in aggregate_functions:
            raise ValueError(f"{val} is not a valid aggregate function")
        return  val


class TableColumns(BaseConfig):
    """
    A table and list of columns to get from that table.
    """

    table_name: str
    columns: list[Column] = "All"


class JoinInfo(BaseConfig):
    """
    Info to form a join in a sql statement.
    """

    left_table: str
    right_table: str
    left_column: str
    right_column: str
    how: str = "inner"


class WhereInfo(BaseConfig):
    """
    Info to form a WHERE statement in a query.
    """

    table: str
    column: str
    operator: str
    match: Union[str, int, list]


class MainSqlConf(BaseConfig):
    file: Path
    tables: list[TableColumns]
    joins: list[JoinInfo]  = None
    wheres: list[WhereInfo] = None
    groups: list[TableColumns] = None


    @property
    def conn(self):
        return pyodbc.connect(f"DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={self.file}")


def connection(file):
    return pyodbc.connect(f"DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={file}")


class QueryBuilder:
    """
    Class for building a sql query and returning a dataframe.
    """

    def __init__(
        self,
        params: MainSqlConf,
    ):
        self.cursor = params.conn.cursor()
        self.tables = params.tables
        self.joins = params.joins
        self.where = params.wheres
        self.group = params.groups

    @property
    def table_info(self):
        tables = [table.table_name for table in self.cursor.tables(tableType="TABLE")]
        file_info = {}
        for tab_name in tables:
            cols = self.cursor.columns(table=tab_name)
            table_info = {}
            for col in cols:
                table_info[col.column_name] = col.type_name
            file_info[tab_name] = table_info
        return file_info

    @property
    def select_string(self):
        table_strings = []
        # if self.group is not None:
        for table in self.tables:
            if table.columns == "All":
                string = f"{table.table_name}.*"
            else:
                string = ", ".join(
                    [
                        f"{column.groupby_fn}([{table.table_name}].[{column.column_name}])"
                        if column.groupby_fn is not None
                        else f"[{table.table_name}].[{column.column_name}]"
                        for column in table.columns
                    ]
                )
            table_strings.append(string)
        return ", ".join(table_strings)

    @property
    def join_string(self):
        join_strings = []
        if self.joins:
            for join in self.joins:
                join_string = f"""{join.how.upper()} JOIN {join.right_table} ON {join.left_table}.{join.left_column} = {join.right_table}.{join.right_column})"""
                join_strings.append(join_string)
            join_strings.insert(0, self.joins[0].left_table)
            return "(" * len(self.joins) + " ".join(join_strings)
        else:
            return  self.tables[0].table_name

    @property
    def where_string(self):
        where_strings = []
        for where in self.where:
            if isinstance(where.match, list):
                if self.table_info[where.table][where.column] in STRINGTYPENAMES:
                    lis = [f"'{item}'" for item in where.match]
                else:
                    lis = [str(item) for item in where.match]
                match_string = f"({', '.join(lis)})"
            else:
                if self.table_info[where.table][where.column] in STRINGTYPENAMES:
                    match_string = f"'{where.match}'"
                else:
                    match_string = where.match
            where_strings.append(
                f"[{where.table}].[{where.column}] {where.operator} {match_string}"
            )
        return f'\nWHERE {" AND ".join(where_strings)}'

    @property
    def group_string(self):
        table_strings = []
        for table in self.group:
            string = ", ".join(
                f"[{table.table_name}].[{column.column_name}]" for column in table.columns
            )
            table_strings.append(string)
        return ", ".join(table_strings) + ';'

    def load_db(self):
        query = f"""SELECT {self.select_string}
FROM {self.join_string}"""
        if self.where is not None:
            query += self.where_string
        if self.group is not None:
            query += f"\nGROUP BY {self.group_string}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        columns = []
        for table in self.tables:
            if table.columns == "All":
                columns += [
                    column.column_name
                    for column in self.cursor.columns(table=table.table_name)
                ]
            else:
                for name in table.columns:
                    columns.append(name.column_name)
        df = pd.DataFrame([tuple(row) for row in rows], columns=columns)
        if self.group is not None:
            index_cols = []
            for table in self.group:
                for name in table.columns:
                    index_cols.append(name.column_name)
            df.set_index(index_cols, inplace=True)
        return df

