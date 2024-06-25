from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import ChatBot

# system prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a helpful assistant. Answer all questions to the best of your ability.',
        ),
        # Human message shall in the format {'messages': [HumanMessage(), ...]}
        MessagesPlaceholder(variable_name='messages'),
    ]
)

chatbot = ChatBot(prompt)

while True:
    user_input = input('You: ')
    if user_input == '.exit':
        print('bye bye')
        break
    chatbot.chat(user_input)
