# Application of LLM uses

## Retrieval-Augmented Generation (RAG)
RAG combines the capabilities of retrieval-based systems and generative models to provide more accurate and context-aware responses.

1. [How RAG works](./RAG.md)

2. [Fine-tuning RAG](./RAGTuning.md)

3. [Graph RAG](https://github.com/microsoft/graphrag) - knowledge graph with the neo4j database

## GPT Researcher
GPT Researcher is an autonomous agent designed for comprehensive online research on a variety of tasks. [Example](./agent-gpt-researcher.py)

https://github.com/assafelovic/gpt-researcher?tab=readme-ov-file


## Open Interpreter
Open Interpreter lets language models run code. You can chat with Open Interpreter through a ChatGPT-like interface in your terminal. This provides a natural-language interface to your computer’s general-purpose capabilities.

https://github.com/OpenInterpreter/open-interpreter

## LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models
LIDA is a library for generating data visualizations and data-faithful infographics.

https://github.com/microsoft/lida

## AutoGen
AutoGen is a framework that enables the development of LLM applications using multiple agents that can converse with each other to solve tasks.

https://github.com/microsoft/autogen

## LangChain

1. [Financial Agent with LangGraph](./agent-financial.py) 
    + Integrate a third party API (i.e. [polygon.io](https://polygon.io/)) as Tool(s)
    + Use [LangGraph](https://langchain-ai.github.io/langgraphjs/tutorials/) to costruct a cyclic loop between agent and its tools

2. SQL Agent - a ReAct(Reasoning and Acting) Agent
    + [SQL Agent](./agent-sql.py)
    + [Prompt](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/mrkl/prompt.py)