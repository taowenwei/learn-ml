from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class MongoChat:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    '''You are a mongoDB expert. 
                    
                    Only when you are asked, you produce query or aggregation source code. 
                    
                    Generate only source code, no explaination''',
                ),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )
        model = ChatOpenAI(model='gpt-3.5-turbo')
        self.config = {'configurable': {'session_id': 'any'}}
        self.chain = RunnableWithMessageHistory(
            prompt | model, get_session_history)

    def addSchema(self, collectionName, schemaJson):
        mdMsg = f'''
            the following is the `{collectionName}` collection's schema in json format,

            ```json
            {schemaJson}
            ```

            do not generate a response
            '''
        self.getAIMessage(HumanMessage(content=mdMsg))

    def addKnowledge(self, knowledge):
        self.getAIMessage(HumanMessage(
            content=f'''take it as a fact, `{knowledge}`. 
            
            do not generate a response'''))

    def askQuestion(self, question):
        self.getAIMessage(HumanMessage(
            content=f'answer the question, {question}'))

    def getAIMessage(self, humanMsg):
        for chunk in self.chain.stream(
                {'messages': [humanMsg]}, self.config):
            print(chunk.content, end='')
