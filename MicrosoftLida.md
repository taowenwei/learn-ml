# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models

## Setup for debugging
1. Export an OpenAI API key
```bash
export OPENAI_API_KEY=YOUR_API_KEY
```
2. Following the [instructions](./UvicornDebugging.md) to prepare for debugging
3. Run the `nvicron` app directly from command line
```bash
uvicorn lida.web.app:app --host 127.0.0.1 --port 8080 --reload
```
4. Launch (by attaching) the server app debugging from VS code

## Architecture
Lida is a full stack app. It takes advantage of the `nvicron` app to host the frontend and spawns a backend at `lida/web/app.py` (hence the above debugging setup)

It imports a pandas dataframe from either a json or a csv file. Then utilize the followings to generate a data summary and  visualizations with python source code provided.

    1. a LLM engine
    2. a set of system prompts

