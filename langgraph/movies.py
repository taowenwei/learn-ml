from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
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


@tool
def get_movies_movies_get(year: Optional[int] = None) -> str:
    '''Get all movies or movies by a release year'''

    url = 'http://localhost:4000/movies'
    if year != None:
        url += f'?year={year}'
    response = requests.get(url)
    return response.json()


@tool
def get_movie_by_id_movies__id__get(id: int) -> dict:
    '''Get movie by Id'''

    url = f'http://localhost:4000/movies/{id}'
    response = requests.get(url)
    return response.json()


@tool
def get_movie_years_movies_years__get(year: Optional[int] = None) -> list[int]:
    '''Get all moovie release years'''

    url = 'http://localhost:4000/movies/years/'
    response = requests.get(url)
    return response.json()


class Assistant:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                'You are a specialized assistant for movie records. You can, '
                '1. use the `get_movies_movies_get` tool to get all movies or all movies of a release year'
                '2. use the `get_movie_by_id_movies__id__get` tool to get a movie by its id'
                '3. use the `get_movie_years_movies_years__get` tool to get all movie release years'
            ),
            ('user', '{user}'),
        ])

    tools = [get_movies_movies_get,
             get_movie_by_id_movies__id__get,
             get_movie_years_movies_years__get]

    def __init__(self):
        llm = ChatOpenAI(model='gpt-3.5-turbo')
        self.runnable = Assistant.prompt | llm.bind_tools(Assistant.tools)

    def __call__(self, state: utils.State, config: RunnableConfig):
        return {'messages': [self.runnable.invoke(state['messages'])]}


builder = StateGraph(utils.State)
builder.add_node('movieAssistant', Assistant())
builder.add_node('tools', utils.toolNodeWithFallback(Assistant.tools))
builder.add_conditional_edges(
    'movieAssistant',
    tools_condition,
)
builder.add_edge('tools', 'movieAssistant')
builder.set_entry_point('movieAssistant')
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(':memory:'))

# utils.graph2png(graph)

user_input = 'can you get me all the movies released on 2008?'
config = {'configurable': {'thread_id': '1'}}
utils.graphStream(graph, config, user_input)
