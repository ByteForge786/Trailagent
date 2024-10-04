import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, initialize_agent
from langchain.llms.base import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from toolkit import AgentToolkit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeCortexLLM(LLM):
    conn: Any

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{prompt}');"
        logger.info(f"Executing Cortex query: {query}")
        result = pd.read_sql(query, self.conn)
        logger.info(f"Cortex inference result: {result.iloc[0, 0]}")
        return result.iloc[0, 0]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name": "SnowflakeCortexLLM"}

    @property
    def _llm_type(self) -> str:
        return "snowflake_cortex"

class Agent:
    agent_executor: AgentExecutor

    def __init__(self, conn: Any):
        logger.info("Initializing Agent")
        snowflake_llm = SnowflakeCortexLLM(conn=conn)
        toolkit = AgentToolkit(llm=snowflake_llm, conn=conn)
        tools = toolkit.get_tools()

        system_message = """
        You are a helpful assistant for analyzing and optimizing queries running on Snowflake to reduce resource consumption and improve performance.
        If the user's question is not related to query analysis or optimization, then politely refuse to answer it.
        Scope: Only analyze and optimize SELECT queries. Do not run any queries that mutate the data warehouse (e.g., CREATE, UPDATE, DELETE, DROP).
        YOU SHOULD FOLLOW THIS PLAN and seek approval from the user at every step before proceeding further:
        1. Identify Expensive Queries
            - For a given date range (default: last 7 days), identify the top 20 most expensive `SELECT` queries using the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
            - Criteria for "most expensive" can be based on execution time or data scanned.
        2. Analyze Query Structure
            - For each identified query, determine the tables being referenced in it and then get the schemas of these tables to under their structure.
        3. Suggest Optimizations
            - With the above context in mind, analyze the query logic to identify potential improvements.
            - Provide clear reasoning for each suggested optimization, specifying which metric (e.g., execution time, data scanned) the optimization aims to improve.
        4. Validate Improvements
            - Run the original and optimized queries to compare performance metrics.
            - Ensure the output data of the optimized query matches the original query to verify correctness.
            - Compare key metrics such as execution time and data scanned, using the query_id obtained from running the queries and the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
        5. Prepare Summary
            - Document the approach and methodology used for analyzing and optimizing the queries.
            - Summarize the results, including:
                - Original vs. optimized query performance
                - Metrics improved
                - Any notable observations or recommendations for further action
        """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        logger.info("Initializing agent")
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=snowflake_llm,
            agent="zero-shot-react-description",
            verbose=True,
            agent_kwargs={"system_message": system_message}
        )
        logger.info("Agent initialization complete")
    
    def get_executor(self) -> AgentExecutor:
        return self.agent_executor
