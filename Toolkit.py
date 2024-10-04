import logging
from typing import List, Optional, Type, Sequence, Dict, Any, Union, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )

class InfoSnowflakeTableTool(BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    conn: Any = Field(exclude=True)

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        logger.info(f"Getting schema for tables: {table_names}")
        output_schema = ""
        _table_names = table_names.split(",")
        for t in _table_names:
            query = f"DESCRIBE TABLE {t}"
            logger.info(f"Executing query: {query}")
            schema = pd.read_sql(query, self.conn)
            logger.info(f"Schema for table {t}:\n{schema.to_string()}")
            output_schema += f"Schema for table {t}:\n{schema.to_string()}\n\n"
        return output_schema

class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")

class QuerySQLCheckerTool(BaseTool):
    """Uses Snowflake Arctic model to check if a query is correct."""

    template: str = """
    {query}
    Double check the {dialect} query above for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    Output the final SQL query only.

    SQL Query: """
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    conn: Any = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Snowflake Cortex to check the query."""
        logger.info(f"Checking query: {query}")
        escaped_query = query.replace('"', '\\"').replace("'", "\\'")
        prompt = self.template.format(query=escaped_query, dialect="Snowflake")
        cortex_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{prompt}');"
        logger.info(f"Executing Cortex query: {cortex_query}")
        result = pd.read_sql(cortex_query, self.conn)
        checked_query = result.iloc[0, 0]
        logger.info(f"Checked query result: {checked_query}")
        return checked_query

class _QuerySQLDataBaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")

class QuerySQLDataBaseTool(BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result and query_id.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    conn: Any = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
        """Execute the query, return the results and query_id; or an error message."""
        logger.info(f"Executing query: {query}")
        try:
            results = pd.read_sql(query, self.conn)
            logger.info(f"Query results:\n{results.to_string()}")
            cursor = self.conn.cursor()
            cursor.execute(query)
            query_id = cursor.sfqid
            cursor.close()
            logger.info(f"Query ID: {query_id}")
            return results, query_id
        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error(error_msg)
            return error_msg, None

class AgentToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    conn: Any = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return "Snowflake"

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        logger.info("Initializing AgentToolkit tools")
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSnowflakeTableTool(
            conn=self.conn, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result and query_id from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            conn=self.conn, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            conn=self.conn, description=query_sql_checker_tool_description
        )
        logger.info("AgentToolkit tools initialized")
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            query_sql_checker_tool,
        ]
