from IPython.display import Image, display


def graph2png(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


def graphStream(graph, config, inputMsg=None):
    events = graph.stream({"messages": [(
        "user", inputMsg)]} if inputMsg != None else None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
