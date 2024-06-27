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


class Assistant:
    BaseUrl = 'http://localhost:4000'

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                '''You are a specialized assistant for movie records. You can,
                1. use the `get_movies_movies_get` tool to get all movies or all movies of a release year
                2. use the `get_movie_by_id_movies__id__get` tool to get a movie by its id
                3. use the `get_movie_years_movies_years__get` tool to get all movie release years
                
                Now answer your question'''
            ),
            ('user', '{user}'),
        ])

    @tool
    def get_movies_movies_get(year: Optional[int] = None) -> str:
        '''Get all movies or movies by a release year'''

        url = f'{Assistant.BaseUrl}/movies' + \
            (f'?year={year}' if year != None else '')
        response = requests.get(url)
        return response.json()

    @tool
    def get_movie_by_id_movies__id__get(id: int) -> dict:
        '''Get movie by Id'''

        response = requests.get(f'{Assistant.BaseUrl}/movies/{id}')
        return response.json()

    @tool
    def get_movie_years_movies_years__get(year: Optional[int] = None) -> list[int]:
        '''Get all moovie release years'''

        response = requests.get(f'{Assistant.BaseUrl}/movies/years/')
        return response.json()

    @tool
    def create_movie_movies_post(title: str, year: int) -> dict:
        '''Create a new movie'''

        response = requests.post(f'{Assistant.BaseUrl}/movies', json={
            'title': title,
            'year': year
        })
        return response.json()

    tools = [get_movies_movies_get,
             get_movie_by_id_movies__id__get,
             get_movie_years_movies_years__get,
             create_movie_movies_post]

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

config = {'configurable': {'thread_id': '1'}}
# user_input = 'can you get me all the movies released on 2008?'
# utils.graphStream(graph, config, user_input)
# user_input = 'can you get me all the movies\' release years?'
# utils.graphStream(graph, config, user_input)
# user_input = 'can you get me the movie with id=2?'
# utils.graphStream(graph, config, user_input)
user_input = 'can you add a new movie with title="Despicable Me 4", year=2024?'
utils.graphStream(graph, config, user_input)

