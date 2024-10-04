import snowflake.connector
import pandas as pd
from typing import Dict, Any, List, Optional
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Your existing cortex_inference function
def cortex_inference(prompt: str) -> str:
    query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', '{prompt}');"
    
    # Assuming you have a working Snowflake connection
    con = snowflake.connector.connect(
        # Connection params
    )
    
    result = pd.read_sql(query, con)
    con.close()
    return result.iloc[0, 0]

# Custom LLM class for Snowflake Cortex
class SnowflakeCortexLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return cortex_inference(prompt)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name": "SnowflakeCortexLLM"}

    @property
    def _llm_type(self) -> str:
        return "snowflake_cortex"

# Initialize the Snowflake Cortex LLM
snowflake_llm = SnowflakeCortexLLM()

# Define a simple tool that the agent can use
def get_word_length(word: str) -> int:
    return len(word)

# Create a LangChain Tool from the function
word_length_tool = Tool(
    name="WordLength",
    func=get_word_length,
    description="Useful for getting the length of a word"
)

# Create a simple prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Answer the following question: {query}"
)

# Create an LLMChain
llm_chain = LLMChain(llm=snowflake_llm, prompt=prompt_template)

# Initialize the agent
agent = initialize_agent(
    tools=[word_length_tool],
    llm=snowflake_llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Use the agent
result = agent.run("What's the length of the word 'python'?")
print(result)

# Example of using the LLMChain directly
chain_result = llm_chain.run("Explain what Python is in one sentence.")
print(chain_result)
