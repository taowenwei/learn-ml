from IPython.display import Image
import multiprocessing
from io import BytesIO
import PIL
import PIL.Image
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from typing_extensions import TypedDict

# Global state with `messages` as history
# `messages` will be referenced in graphStream & toolNodeWithFallback

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def graphStream(graph, config, inputMsg=None):
    events = graph.stream({"messages": [(
        "user", inputMsg)]} if inputMsg != None else None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def toolNodeWithFallback(tools: list) -> dict:
    def handleToolError(state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handleToolError)], exception_key="error"
    )


def runCode(code_string):
    exec(code_string)


def pythonProcess(code):
    process = multiprocessing.Process(target=runCode, args=(code,))
    # Start the process
    process.start()
    # Wait for the process to complete
    process.join()


def graph2png(graph):
    try:
        img = Image(graph.get_graph().draw_mermaid_png())
        pilImg = PIL.Image.open(BytesIO(img.data))
        pilImg.show()
    except Exception as e:
        # This requires some extra dependencies and is optional
        print(e)
        pass
