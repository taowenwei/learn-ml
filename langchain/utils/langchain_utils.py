from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.globals import set_debug


# Build an Agent
# https://python.langchain.com/v0.2/docs/tutorials/agents/


class LangChainAgent:
    def __init__(self, tools, systemPrompt=None, timeout=None):
        memory = SqliteSaver.from_conn_string(':memory:')
        model = ChatOpenAI(model='gpt-3.5-turbo')
        self.config = {'configurable': {
            'thread_id': str(datetime.now().timestamp())}}
        # add interrupt before executing a tool
        # interrupt_before = ['tools']
        # most verbose setting with fully log raw inputs and outputs.
        # set_debug(True)
        self.agent = create_react_agent(
            model, tools, messages_modifier=systemPrompt, checkpointer=memory)
        self.agent.step_timeout = timeout

    def execute(self, question):
        for chunk in self.agent.stream(
            {'messages': [HumanMessage(
                content=question)]},
                self.config, stream_mode='values'
        ):
            message = chunk['messages'][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()


class ChatBot:
    def __init__(self, systemPrompt):
        self.history = ChatMessageHistory()

        def get_session_history(session):
            return self.history

        model = ChatOpenAI(model='gpt-3.5-turbo')
        # chain prompt with model
        chain = systemPrompt | model
        # chain with chat history
        self.chatbot = RunnableWithMessageHistory(chain, get_session_history)

    def chat(self, humanMsg):
        # stream out llm output as it generates tokens (instead of printing after all tokens are generated)
        for r in self.chatbot.stream(
            {'messages': [HumanMessage(
                content=humanMsg)]},
                config={'configurable': {'session_id': 'session_id'}}):
            print(r.content, end='')
        print()
