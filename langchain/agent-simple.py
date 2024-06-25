
from langchain_core.tools import StructuredTool, ToolException
from utils import LangChainAgent

# Build an Agent
# https://python.langchain.com/v0.2/docs/tutorials/agents/

def get_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    confirm = input('Are you sure to execute the query (yes/no)? ')
    if confirm.lower() == 'yes':
        return f'It\'s always sunny in {location}'
    raise ToolException(f"User aborted")


# add tool - https://python.langchain.com/v0.2/docs/how_to/custom_tools/
tools = [StructuredTool.from_function(
    func=get_weather, handle_tool_error=True, )]
# add simple prompt
system_prompt = 'You are a helpful bot named Fred. Only use the `get-weather` tool when weather is asked'
agent = LangChainAgent(tools, system_prompt)

agent.execute('my name is wenwei. i live in san jose, CA')
agent.execute('whats the weather in my town?')
agent.execute('whats my name?')
