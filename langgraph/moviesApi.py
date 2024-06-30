from typing import Optional
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


class MoviesApi:

    BaseUrl = 'http://localhost:4000'

    HttpHeader = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_ACCESS_TOKEN'
        }

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '''You are a specialized manager movie records. You can,

            + use the `get_movies_movies_get` tool to get all movies or all movies of a given release year
            + use the `create_movie_movies_post` tool to get all movie release years
            + use the `get_movie_by_id_movies__id__get` tool to get a movie by its id
            + use the `get_movie_by_id_comment_by_cid_movies__id__comments__cid__get` tool to from a movie of id, get one of its comments of cid
            + use the `get_movie_years_movies_years__get` tool to get all movie release years
            
            Now answer your question'''
        ),
        ('user', '{user}'),
    ])

    @tool
    def get_movies_movies_get(year: Optional[int], sort: Optional[bool]) -> list:
        '''Get all movies or all movies of a given release year'''
        
        response = requests.get(MoviesApi.BaseUrl + '/movies?' + (f'year={year}' if year != None else '') + '&' + (f'sort={sort}' if sort != None else ''), headers=MoviesApi.HttpHeader)
        if response.status_code == 200:
            return 'Successful Response' + '\n\n' + json.dumps(response.json(), indent = 2)
        elif response.status_code == 422:
            return 'Validation Error' + '\n\n' + json.dumps(response.json(), indent = 2)
        return f'Request failed with status code: {response.status_code}'
            
    @tool
    def create_movie_movies_post(title: str, year: int) -> dict:
        '''Get all movie release years'''
        
        data = {
            'title': title, 
            'year': year, 
        }
        response = requests.post(MoviesApi.BaseUrl + '/movies', headers=MoviesApi.HttpHeader, json=data)
        if response.status_code == 200:
            return 'Successful Response' + '\n\n' + json.dumps(response.json(), indent = 2)
        elif response.status_code == 422:
            return 'Validation Error' + '\n\n' + json.dumps(response.json(), indent = 2)
        return f'Request failed with status code: {response.status_code}'
            
    @tool
    def get_movie_by_id_movies__id__get(id: int) -> dict:
        '''Get a movie by its Id'''
        
        response = requests.get(MoviesApi.BaseUrl + '/movies/{id}', headers=MoviesApi.HttpHeader)
        if response.status_code == 200:
            return 'Successful Response' + '\n\n' + json.dumps(response.json(), indent = 2)
        elif response.status_code == 422:
            return 'Validation Error' + '\n\n' + json.dumps(response.json(), indent = 2)
        return f'Request failed with status code: {response.status_code}'
            
    @tool
    def get_movie_by_id_comment_by_cid_movies__id__comments__cid__get(id: int, cid: int) -> dict:
        '''From a movie of id, get one of its comments of cid'''
        
        response = requests.get(MoviesApi.BaseUrl + '/movies/{id}/comments/{cid}', headers=MoviesApi.HttpHeader)
        if response.status_code == 200:
            return 'Successful Response' + '\n\n' + json.dumps(response.json(), indent = 2)
        elif response.status_code == 422:
            return 'Validation Error' + '\n\n' + json.dumps(response.json(), indent = 2)
        return f'Request failed with status code: {response.status_code}'
            
    @tool
    def get_movie_years_movies_years__get() -> list:
        '''Get all movie release years'''
        
        response = requests.get(MoviesApi.BaseUrl + '/movies/years/', headers=MoviesApi.HttpHeader)
        if response.status_code == 200:
            return 'Successful Response' + '\n\n' + json.dumps(response.json(), indent = 2)
        return f'Request failed with status code: {response.status_code}'
            

    tools = [
        get_movies_movies_get,
        create_movie_movies_post,
        get_movie_by_id_movies__id__get,
        get_movie_by_id_comment_by_cid_movies__id__comments__cid__get,
        get_movie_years_movies_years__get,
    ]

    def __init__(self, llm):
        self.runnable = MoviesApi.prompt | llm.bind_tools(MoviesApi.tools)

    def __call__(self, state, config: RunnableConfig):
        return {'messages': [self.runnable.invoke(state['messages'])]}
    