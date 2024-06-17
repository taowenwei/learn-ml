#  from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_openai.chat_models import ChatOpenAI


# llm = Ollama(model="llama3")
llm = ChatOpenAI(model="gpt-4-0125-preview")

db = SQLDatabase.from_uri("sqlite:///example.db", sample_rows_in_table_info = 3)
db.get_usable_table_names()
print(db.table_info)

from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

agent_executor = create_sql_agent(llm, db = db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
agent_executor.invoke("How many users are in the database?")