from typing import Optional
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import requests
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import utils
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig


from typing import Optional
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import moviesApi


llm = ChatOpenAI(model='gpt-3.5-turbo')

builder = StateGraph(utils.State)
builder.add_node('movieAssistant', moviesApi.MoviesApi(llm))
builder.add_node('tools', utils.toolNodeWithFallback(
    moviesApi.MoviesApi.tools))
builder.add_conditional_edges(
    'movieAssistant',
    tools_condition,
)
builder.add_edge('tools', 'movieAssistant')
builder.set_entry_point('movieAssistant')
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(':memory:'))

# utils.graph2png(graph)

config = {'configurable': {'thread_id': '1'}}
while True:
    user_input = input('You: ')
    if user_input == '.exit':
        print('bye bye')
        break
    utils.graphStream(graph, config, user_input)


# user_input = 'can you get me all the movies released on 2008?'
# user_input = 'can you get me all the movies\' release years?'
# user_input = 'can you get me the movie with id=2?'
# user_input = 'can you add a new movie with title="Despicable Me 4", year=2024?'
