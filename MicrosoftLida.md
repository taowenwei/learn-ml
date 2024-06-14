# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models

## Setup for debugging
1. Export OpenAI API key
```bash
export OPENAI_API_KEY=YOUR_API_KEY
```
2. Following [instructions](./UvicornDebugging.md) to set up for debugging
3. run the `nvicron` app directly
```bash
uvicorn lida.web.app:app --host 127.0.0.1 --port 8080 --reload
```
4. VS code launch the server app debugging

## Architecture
Lida is a full stack app. It takes advantage of the `nvicron` app to host the frontend and spawns a backend (hence the above debugging setup)

It imports a pandas dataframe from either a json or a csv file. Then utilize the followings,

    1. a LLM engine
    2. a set of system prompts

 to generate a data summary and  visualizations with python source code provided.