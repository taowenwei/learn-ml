To attach VSCode to a running Python process for debugging, you need to set up the Python application with the necessary debugging hooks and then configure VSCode to attach to that process. Here’s a step-by-step guide to achieve this:

### Step 1: Install ptvsd (Python Tools for Visual Studio Debugger)

You will need `ptvsd` (or `debugpy`, which is the successor to `ptvsd`). Install `debugpy` using pip:

```bash
pip install debugpy
```

### Step 2: Modify Your Python Script to Enable Debugging

Modify your Python script to enable the debug server. This example assumes you have a FastAPI application, but it works for any Python script.

**app/main.py**:
```python
import debugpy
from fastapi import FastAPI

# Allow other computers to attach to debugpy at this IP address and port.
debugpy.listen(("0.0.0.0", 5678))

# This will block until VS Code is attached and its debugger is ready to begin.
print("Waiting for debugger attach")
debugpy.wait_for_client()
print("Debugger is attached")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### Step 3: Run Your Python Script

Run your Python script as you normally would. For example:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4: Configure VSCode to Attach to the Running Process

1. **Open your project in VSCode**.
2. **Open the Command Palette** by pressing `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
3. Type and select **"Debug: Open launch.json"**. If you don’t have a `launch.json` file yet, VSCode will prompt you to create one.

Add the following configuration to your `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Python",
            "type": "python",
            "request": "attach",
            "host": "localhost",
            "port": 5678,
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",  // Adjust this to your project's root
                    "remoteRoot": "."  // Adjust this if the Python script runs in a different environment
                }
            ]
        }
    ]
}
```

### Step 5: Start the Debugger in VSCode

1. **Set breakpoints** in your code where you want to start debugging.
2. **Run the debugger**:
   - Open the **Run and Debug** pane in VSCode by clicking on the debug icon on the sidebar or pressing `Ctrl+Shift+D`.
   - Select **Attach to Python** from the dropdown and click the **Start Debugging** button (green play button) or press `F5`.

VSCode will attach to the running Python process, and you will be able to hit breakpoints, inspect variables, and perform other debugging tasks.

### Summary

By modifying your Python script to include `debugpy` and configuring VSCode to attach to the running process, you can effectively debug a running Python application. This setup is especially useful for debugging applications that run in long-lived processes, such as web servers or background jobs.