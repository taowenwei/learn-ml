from fastapi import FastAPI, HTTPException
from typing import List, Optional
import uvicorn

app = FastAPI()

# Mock data
movies = [
    {"id": 1, "title": "The Shawshank Redemption", "year": 1994},
    {"id": 2, "title": "The Godfather", "year": 1972},
    {"id": 3, "title": "The Dark Knight", "year": 2008},
    # Add more movies here
]

@app.get("/movies", response_model=List[dict])
def get_movies():
    return movies

@app.get("/movies/{id}", response_model=dict)
def get_movie_by_id(id: int):
    movie = next((movie for movie in movies if movie["id"] == id), None)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.get("/years", response_model=List[int])
def get_movie_years():
    return list(set(movie["year"] for movie in movies))

@app.get("/movies/years/{year}", response_model=List[dict])
def get_movies_by_year(year: int):
    movies_by_year = [movie for movie in movies if movie["year"] == year]
    if not movies_by_year:
        raise HTTPException(status_code=404, detail="No movies found for the given year")
    return movies_by_year

@app.post("/movies", response_model=dict)
def create_movie(title: str, year: int):
    movie_id = len(movies) + 1
    new_movie = {"id": movie_id, "title": title, "year": year}
    movies.append(new_movie)
    return new_movie

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)