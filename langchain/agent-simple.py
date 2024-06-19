from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI


def get_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    confirm = input('Are you sure to execute the query (yes/no)? ')
    if confirm.lower() == 'yes':
        return f'It\'s always sunny in {location}'
    raise ToolException(f"User aborted")


def execute(agent, question, config):
    for chunk in agent.stream(
        {'messages': [HumanMessage(
            content=question)]},
            config, stream_mode='values'
    ):
        message = chunk['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# add tool
tools = [StructuredTool.from_function(
    func=get_weather, handle_tool_error=True)]
model = ChatOpenAI(model='gpt-3.5-turbo')
# add memory
memory = SqliteSaver.from_conn_string(':memory:')
config = {'configurable': {'thread_id': '1234'}}
# add simple prompt
system_prompt = 'You are a helpful bot named Fred. Only use the `get-weather` tool when weather is asked'
# add interrupt before executing a tool
# interrupt_before = ['tools']

agent = create_react_agent(
    model, tools,  # interrupt_before=interrupt_before,
    messages_modifier=system_prompt, checkpointer=memory)
# add 10 sec timeout
agent.step_timeout = 10
execute(agent, 'my name is wenwei. i live in san jose, CA', config)
execute(agent, 'whats the weather in my town?', config)
execute(agent, 'whats my name?', config)
