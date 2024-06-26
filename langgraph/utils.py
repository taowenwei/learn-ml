from IPython.display import Image
import multiprocessing
from io import BytesIO
import PIL
import PIL.Image

def graph2png(graph):
    try:
        img = Image(graph.get_graph().draw_mermaid_png())
        pilImg = PIL.Image.open(BytesIO(img.data))
        pilImg.show()
    except Exception as e:
        # This requires some extra dependencies and is optional
        print(e)
        pass


def graphStream(graph, config, inputMsg=None):
    events = graph.stream({"messages": [(
        "user", inputMsg)]} if inputMsg != None else None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def runCode(code_string):
    exec(code_string)


def pythonProcess(code):
    process = multiprocessing.Process(target=runCode, args=(code,))
    # Start the process
    process.start()
    # Wait for the process to complete
    process.join()
