from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import requests
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import utils
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


@tool
def get_movies_movies_get(year: Optional[int] = None) -> str:
    """Get all movies or movies by a release year"""

    url = 'http://localhost:4000/movies'
    if year != None:
        url += f'?year={year}'
    response = requests.get(url)
    return response.json()


@tool
def get_movie_by_id_movies__id__get(id: int) -> dict:
    """Get movie by Id"""

    url = f'http://localhost:4000/movies/{id}'
    response = requests.get(url)
    return response.json()


@tool
def get_movie_years_movies_years__get(year: Optional[int] = None) -> list[int]:
    """Get all moovie release years"""

    url = 'http://localhost:4000/movies/years/'
    response = requests.get(url)
    return response.json()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for movie records. You can, "
            "1. use the `get_movies_movies_get` tool to get all movies or all movies of a release year"
            "2. use the `get_movie_by_id_movies__id__get` tool to get a movie by its id"
            "3. use the `get_movie_years_movies_years__get` tool to get all movie release years"
        ),
        ("placeholder", "{messages}"),
    ])

tools = [get_movies_movies_get,
         get_movie_by_id_movies__id__get,
         get_movie_years_movies_years__get]
llm = ChatOpenAI(model='gpt-3.5-turbo')
chatbotRunnable = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [chatbotRunnable.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=tools))
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
builder.add_edge("tools", "chatbot")
builder.set_entry_point("chatbot")
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

utils.graph2png(graph)

user_input = "can you get me all the movies released on 2008?"
config = {"configurable": {"thread_id": "1"}}
utils.graphStream(graph, config, user_input)
