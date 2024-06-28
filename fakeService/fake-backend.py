from fastapi import FastAPI, Request, HTTPException, status
from typing import List, Optional
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
import json

app = FastAPI(title='manager of movie records')

# Mock data
movies = [
    {"id": 1, "title": "The Shawshank Redemption", "year": 1994},
    {"id": 2, "title": "The Godfather", "year": 1972},
    {"id": 3, "title": "The Dark Knight", "year": 2008},
    # Add more movies here
]


class Movie(BaseModel):
    id: int = None
    title: str
    year: int


@app.get("/movies", response_model=List[Movie], summary="Get all movies or all movies of a given release year")
def get_movies(year: int = None):
    if year != None:
        movies_by_year = [movie for movie in movies if movie["year"] == year]
        if not movies_by_year:
            raise HTTPException(
                status_code=404, detail="No movies found for the given year")
        return movies_by_year
    return movies


@app.get("/movies/{id}", response_model=Movie, summary='Get a movie by its Id')
def get_movie_by_id(id: int):
    movie = next((movie for movie in movies if movie["id"] == id), None)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@app.get("/movies/years/", response_model=List[int], summary='Get all movie release years')
def get_movie_years():
    return list(set(movie["year"] for movie in movies))


@app.post("/movies", response_model=Movie, summary='Get all movie release years')
def create_movie(movie: Movie):
    movie_id = len(movies) + 1
    new_movie = {"id": movie_id, "title": movie.title, "year": movie.year}
    movies.append(new_movie)
    return new_movie

def toOpenAPISpec():
    with open('openapi.json', 'w') as f:
        json.dump(get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        ), f)

# http://127.0.0.1:4000/openapi.json for Open API spec
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
