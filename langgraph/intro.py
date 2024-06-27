from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import utils


tools = [TavilySearchResults(max_results=2)]
llm = ChatOpenAI(model='gpt-3.5-turbo')
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: utils.State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(utils.State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=tools))
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
builder.add_edge("tools", "chatbot")
builder.set_entry_point("chatbot")
graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string(":memory:"),
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)

# utils.graph2png(graph)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
utils.graphStream(graph, config, user_input)

snapshot = graph.get_state(config)
# interrupt_before
action = snapshot.next[0]
if action == 'tools':
    existing_message = snapshot.values["messages"][-1]
    # get tool call details
    existing_message.tool_calls

    user_input = input(
        "Do you approve of the above actions? Type 'y' to continue "
    )
    if user_input.strip() == "y":
        # `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
        utils.graphStream(graph, config)
