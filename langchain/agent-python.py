from langchain import hub
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


def agentExecutorImpl(question):
    from langchain.agents import AgentExecutor
    from langchain.agents import create_openai_functions_agent
    from langchain_experimental.tools import PythonREPLTool

    tools = [PythonREPLTool()]

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_openai_functions_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": question})


def reactAgentImpl(question):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import StructuredTool

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

    def run_python(query: str) -> str:
        '''Return the python code for the specified query.'''
        return 0

    tools = [StructuredTool.from_function(
        func=run_python, handle_tool_error=True, )]

    memory = SqliteSaver.from_conn_string(':memory:')
    config = {'configurable': {'thread_id': '1234'}}
    system_prompt = '''
        Following the steps to answer a question,
        1. generate a python code solution based on the question.
        2. run the `run_python` tool. 
        3. ALWAYS use the tool result as the final answer
        '''

    agent = create_react_agent(
        model, tools, messages_modifier=system_prompt, checkpointer=memory)
    execute(agent, question, config)


question = "What is the 10th fibonacci number?"
# agentExecutorImpl(question)
reactAgentImpl(question)
