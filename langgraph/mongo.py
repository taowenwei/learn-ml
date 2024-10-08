from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import os
from pymongo import MongoClient
import certifi
import json
import utils


class MongoAssistant:

    databaseName = 'YOUR_DB_NAME'

    connectionString = 'YOUR_DB_CONNECT_STRING'

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                '''You are a mongoDB specialist. 
                
                You always use aggregration for query. 
                
                You have the `mongo_aggregate` tool to perform a mongoDB aggregation

                Now answer your question and output a query result as JSON.
            
                If you receive an error from the tools, apologizes and ask for a next question'''
            ),
            ("placeholder", "{messages}"),
        ])

    @classmethod
    def connect(cls):
        client = MongoClient(MongoAssistant.connectionString,
                             tlsCAFile=certifi.where())
        return client[MongoAssistant.databaseName]

    @classmethod
    def humanIntervene(cls, collectionName: str, query: object):
        print(
            f'The following query will be used with the {collectionName} collection,')
        print("'''json")
        print(json.dumps(query, indent=2))
        print("'''")
        print()
        userInput = input('Are you sure to contine (y/n) ')
        if userInput != 'y':
            raise ValueError("User rejected")

    @tool
    def mongo_aggregate(collectionName: str, aggregation: object) -> list:
        '''Perfom a simple mongodb aggregation'''

        database = MongoAssistant.connect()
        collection = database[collectionName]
        MongoAssistant.humanIntervene(collectionName, aggregation)
        return list(collection.aggregate(aggregation))

    tools = [
        mongo_aggregate,
    ]

    def __init__(self, llm):
        self.runnable = MongoAssistant.prompt | llm.bind_tools(
            MongoAssistant.tools)

    def __call__(self, state, config: RunnableConfig):
        configuration = config.get("configurable")
        MongoAssistant.databaseName = configuration.get('name', None)
        MongoAssistant.connectionString = configuration.get('conn', None)
        result = self.runnable.invoke(state)
        return {'messages': result}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# build a LangGraph graph with an agent, tools and memory
llm = ChatOpenAI(model='gpt-4o-mini') #'gpt-3.5-turbo'
builder = StateGraph(State)
builder.add_node('mongoAssistant', MongoAssistant(llm))
builder.add_node('tools', utils.toolNodeWithFallback(MongoAssistant.tools))
builder.add_conditional_edges(
    'mongoAssistant',
    tools_condition,
)
builder.add_edge('tools', 'mongoAssistant')
builder.set_entry_point('mongoAssistant')
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(':memory:'))

# build a conversation thread
config = {'configurable': {'thread_id': '1',
                           # mock server URL and mock API token
                           'conn': os.environ['MONGODB'], 'name': os.environ['MONGO_DB_NAME']}}
while True:
    user_input = input('You: ')
    if user_input == '.exit':
        print('bye bye')
        break

    events = graph.stream(
        {'messages': [('user', user_input)]}, config, stream_mode='values')
    for event in events:
        if 'messages' in event:
            event['messages'][-1].pretty_print()
