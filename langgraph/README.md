# LangGraph Tutorials

## intro
[intro](./intro.py) shows cases,

1. basic chatbot with LangGraph
2. chat with an agent
3. human-in-the-middle before run the agent, by using `interrupt_before`
4. `checkpointer` as chat history

## code assistant
[code assistant](./code-assist.py) shows cases,

1. structured output with schema, by using `llm.with_structured_output`
2. python code generation for execution with `exec`
3. conditional edge for node path routing, by using `add_conditional_edges`

## mongoDB assistant
[mongo assistant](./mongo.py) performs query and aggregation, but is it better to actually export database to csv(s) before-hand and with table joins completed? Benefits,
1. processing csv is way easier
2. after export, data schema is fixed, (perfect for schema-less, Document-based database)
3. after table joins, the semantic of data is aggrgated into one place.

Database agents feel more pertinent to a OLAP situation than a OLTP one.

## Graph RAG

Need to understand knowledge graph first.

## LLM and Agents in General

I believe it shall take examples from a mobile devices and its apps, whereas LLM is the device/the platform, and the agents are installable apps.