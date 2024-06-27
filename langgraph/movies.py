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

class MovieAssistant:
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

    def __init__(self, llm):
        self.assistant = MovieAssistant.prompt | llm.bind_tools(
            MovieAssistant.tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def create_tool_node_with_fallback(tools: list) -> dict:
    def handle_tool_error(state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


memory = SqliteSaver.from_conn_string(":memory:")
movieAssistant = MovieAssistant(ChatOpenAI(model='gpt-3.5-turbo'))
builder = StateGraph(State)
builder.add_node("assistant", movieAssistant.assistant)
builder.add_node("tools", create_tool_node_with_fallback(MovieAssistant.tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=memory)

# utils.graph2png(graph)
