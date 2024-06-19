from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"


def execute(agent, question, config):
    for chunk in agent.stream(
        {"messages": [HumanMessage(
            content=question)]},
            config, stream_mode="values"
    ):
        message = chunk["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


tools = [get_weather]
model = ChatOpenAI(model="gpt-3.5-turbo")
memory = SqliteSaver.from_conn_string(":memory:")
config = {"configurable": {"thread_id": "1234"}}
system_prompt = "You are a helpful bot named Fred. Only use the `get-weather` tool when weather is asked"

agent = create_react_agent(
    model, tools, messages_modifier=system_prompt, checkpointer=memory)
execute(agent, 'my name is wenwei. i live in san jose, CA', config)
execute(agent, 'whats the weather in my town?', config)
execute(agent, 'whats my name?', config)
