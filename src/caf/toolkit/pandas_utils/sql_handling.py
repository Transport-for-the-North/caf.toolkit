from caf.toolkit import BaseConfig
from pathlib import Path
from typing import Union
import pyodbc
import pandas as pd
from pydantic import validator
import enum
import sqlite3

STRINGTYPENAMES = ("VARCHAR", "TEXT", "MEMO", "DATETIME", "YESNO", "CHARACTER")


class AggregateFunctions(enum.Enum):
    """
    Enumeration of valid aggregate function strings for a sql statement.
    """

    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    FIRST = "FIRST"
    LAST = "LAST"
    STDDEV = "STDDEV"
    STDEV_POP = "STDEV_POP"
    VARIANCE = "VARIANCE"
    VAR_POP = "VAR_POP"
    MEDIAN = "MEDIAN"
    GROUP_CONCAT = "GROUP_CONCAT"
    STRING_AGG = "STRING_AGG"
    ANY = "ANY"
    SOME = "SOME"
    ALL = "ALL"
    EVERY = "EVERY"
    RANK = "RANK"
    DENSE_RANK = "DENSE_RANK"
    PERCENT_RANK = "PERCENT_RANK"
    CUME_DIST = "CUME_DIST"
    ROW_NUMBER = "ROW_NUMBER"


class Column(BaseConfig):
    """
    Info about a column of a table, for use in TableColumns.

    Parameters
    ----------

    column_name (str): The name of the column
    groupby_fn: If grouping, the function to groupby for this column.
    """

    column_name: str
    groupby_fn: AggregateFunctions = None

    @property
    def _groupby_fn(self):
        if self.groupby_fn is None:
            return self.groupby_fn
        return self.groupby_fn.value


class TableColumns(BaseConfig):
    """
    A table and list of columns to get from that table.

    Parameters
    ----------

    table_name (str): The name of the table
    columns(list[Column]): A list of columns to extract from that table.
    """

    table_name: str
    columns: list[Column] = "All"


class JoinInfo(BaseConfig):
    """
    Info to form a join in a sql statement.

    Parameters
    ----------

    left_table: The name of the left table in the join
    right_table: The name of the right table in the join
    left_column: The join column in the left table
    right_column: The join column in the right table. This column will be
    dropped post join for inner joins, so do not explicitly use this to group
    on
    how: The join type. Defaults to "inner".
    """

    left_table: str
    right_table: str
    left_column: str
    right_column: str
    how: str = "inner"

    @property
    def join_tuple_tuple(self):
        return ((self.left_table, self.left_column), (self.right_table, self.right_column))


class WhereInfo(BaseConfig):
    """
    Info to form a WHERE statement in a query.

    Parameters
    ----------

    table: The table the column is in
    column: The column being filtered on
    operator: The operator for the where statement (between the column and the
    match)
    match: The match for the column

    Example
    -------
        >>> "WHERE TABLE.COLUMN IN ['FOO','BAR','BAZ']"
        >>> #operator is "IN", match is ['FOO','BAR','BAZ']
    """

    table: str
    column: str
    operator: str
    match: Union[str, int, list]


class MainSqlConf(BaseConfig):
    """
    Main config class for SQL queries.

    Parameters
    ----------

    file: Path to the database file. This can also be a connection to a sqlite
    or access database, but it is expected to receive a path. If a path is
    provided it must be to an access database.
    tables: List of TableColumns, all of the columns you want data from
    joins: List of JoinInfo instances, all of the joins needed. Number of joins
    must be at least number of tables minus 1. Defaults to None.
    wheres: List of WhereInfo instances for all filters needed. Defaults to
    None.
    groups: List of TableColumns for all groups. This should be columns to
    groupby. These should not contain groupby_fn. Defaults to None. If groups
    is not None, then all of the columns in tables which are not grouped on
    MUST have groupby_fn included, or an error will occur.
    """

    file: Union[Path, sqlite3.Connection, pyodbc.Connection]
    tables: list[TableColumns]
    joins: list[JoinInfo] = None
    wheres: list[WhereInfo] = None
    groups: list[TableColumns] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("groups")
    def groupbys(cls, v, values):
        """ """
        # Pairs is a list of tuples of table and column name in groups

        pairs = []
        for table in v:
            for column in table.columns:
                pairs.append((table.table_name, column.column_name))
        # Check joins to add to pairs
        if "joins" in values.keys():
            # For join pairs, if left or right is grouped on, add its partner
            for join in values["joins"]:
                if join.join_tuple_tuple[0] in pairs:
                    pairs.append(join.join_tuple_tuple[1])
                elif join.join_tuple_tuple[1] in pairs:
                    pairs.append(join.join_tuple_tuple[0])

        # Iterate through all of the columns in tables
        for table in values["tables"]:
            # All shouldn't be used if grouping
            if table.columns == "All":
                raise ValueError("All is not a valid option for a table" "when grouping.")
            for column in table.columns:
                # Check that the column isn't being grouped by, and no groupby_fn, raise error
                if ((table.table_name, column.column_name) not in pairs) and (
                    column.groupby_fn is None
                ):
                    raise ValueError(
                        "groups is not None but not all of the "
                        "columns in tables have groupby_fn. "
                        f"{column.column_name} is missing. "
                        "This is the first missing, but not "
                        "necessarily the only one."
                    )
        return v

    @property
    def conn(self):
        if isinstance(self.file, (pyodbc.Connection, sqlite3.Connection)):
            return self.file
        return pyodbc.connect(
            f"DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={self.file}"
        )


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
                        f"{column._groupby_fn}([{table.table_name}].[{column.column_name}])"
                        if column._groupby_fn is not None
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
            return self.tables[0].table_name

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
        return ", ".join(table_strings) + ";"

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
        # Remove duplicated columns from inner joins
        if self.joins is not None:
            join_dropper = []
            for join in self.joins:
                if (join.how == "inner") or (join.how == "left"):
                    join_dropper.append(join.right_column)
                if join.how == "right":
                    join_dropper.append(join.left_column)
            df.drop(join_dropper, axis=1, inplace=True)
        if self.group is not None:
            index_cols = []
            for table in self.group:
                for name in table.columns:
                    index_cols.append(name.column_name)
            df.set_index(index_cols, inplace=True)
        return df
