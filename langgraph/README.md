# LangGraph Tutorials

## intro
[intro](./intro.py) show cases,

1. basic chatbot with LangGraph
2. chat with an agent
3. human-in-the-middle before run the agent, by using `interrupt_before`
4. `checkpointer` as chat history

## code assistant
[code assistant](./code-assist.py) show case,

1. structured output with schema, by using `llm.with_structured_output`
2. python code generation for execution with `exec`
3. conditional edge for node path routing, by using `add_conditional_edges`