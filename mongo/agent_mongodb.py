from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model_with_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "undefined"}}

while True:
    text = input('Human>> ')
    if text == '.exit':
        break
    ai_msg = model_with_history.invoke(
        [HumanMessage(content=text)], config=config)
    print(ai_msg.content)
