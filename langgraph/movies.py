from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import utils
from langchain_openai import ChatOpenAI
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

config = {'recursion_limit': 10,  # fail early
          'configurable': {'thread_id': '1',
                           'url': 'http://localhost:4000', 'token': '1234567890'}}
while True:
    user_input = input('You: ')
    if user_input == '.exit':
        print('bye bye')
        break
    utils.graphStream(graph, config, user_input)


# user_input = 'get all the movies released on 2008'
# user_input = 'retrieve all the movie release years'
# user_input = 'get the movie with id=2'
# user_input = 'create a new movie with title="Despicable Me 4", year=2024'
