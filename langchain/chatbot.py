from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from datetime import datetime

# llm
model = ChatOpenAI(model="gpt-3.5-turbo")

# system prompot
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        # Human message shall in the format {"messages": [HumanMessage(), ...]}
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chat history, supporting multiple sessions
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# chain prompt with model
chain = prompt | model

# chain with chat history
chatbot = RunnableWithMessageHistory(chain, get_session_history)
session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
while True:
    user_input = input("You: ")
    if user_input == '.exit':
        print('bye bye')
        break
    # stream out llm output as it generates tokens (instead of printing after all tokens are generated)
    for r in chatbot.stream(
        {"messages": [HumanMessage(
            content=user_input)]},
            config={"configurable": {"session_id": session_id}}):
        print(r.content, end="")
    print()
print()
