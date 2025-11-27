# codeassistx_mvp_advanced.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import ast, traceback, json, subprocess, uuid, os

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

@app.post("/analyze")
def analyze():
    return {"ok": True}

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeInput(BaseModel):
    code: str

def ast_to_dict(node):
    if not isinstance(node, ast.AST):
        return str(node)
    result = {"_type": node.__class__.__name__}
    for field in node._fields:
        value = getattr(node, field)
        if isinstance(value, list):
            result[field] = [ast_to_dict(v) for v in value]
        elif isinstance(value, ast.AST):
            result[field] = ast_to_dict(value)
        else:
            result[field] = repr(value)
    return result

class CFGNode:
    def __init__(self, nid: int, label: str):
        self.id = nid
        self.label = label
        self.edges = []
    def to_dict(self):
        return {"id": self.id, "label": self.label, "edges": self.edges}

class SimpleCFGGenerator(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.counter = 0
        self.current = None

    def new_node(self, label: str):
        n = CFGNode(self.counter, label)
        self.counter += 1
        self.nodes.append(n)
        return n

    def add_edge(self, a, b):
        if b.id not in a.edges:
            a.edges.append(b.id)

    def generate(self, code: str):
        self.counter = 0
        self.nodes = []
        try:
            tree = ast.parse(code)
        except Exception:
            root = self.new_node("Parse Error")
            end = self.new_node("End")
            self.add_edge(root, end)
            return self.nodes

        entry = self.new_node("Start")
        self.current = entry
        self.visit(tree)
        end = self.new_node("End")
        self.add_edge(self.current, end)
        return self.nodes

    def generic_visit(self, node):
        if isinstance(node, ast.Assign):
            try:
                label = "Assign: " + ", ".join([ast.unparse(t) for t in node.targets]) + " = " + (ast.unparse(node.value) if hasattr(ast, 'unparse') else "")
            except Exception:
                label = "Assign"
            n = self.new_node(label)
            self.add_edge(self.current, n)
            self.current = n
            return
        if isinstance(node, ast.Expr):
            try:
                label = "Expr: " + (ast.unparse(node.value) if hasattr(ast, 'unparse') else "")
            except Exception:
                label = "Expr"
            n = self.new_node(label)
            self.add_edge(self.current, n)
            self.current = n
            return
        super().generic_visit(node)

def safe_exec(code: str):
    logs = []
    env = {}
    try:
        logs.append({"event": "start", "vars": dict(env)})
        exec(compile(code, "<string>", "exec"), {}, env)
        clean = {k: repr(v) for k, v in env.items() if not k.startswith("__")}
        logs.append({"event": "success", "vars": clean})
    except Exception as e:
        logs.append({"event": "error", "error": str(e), "trace": traceback.format_exc()})
    return {"logs": logs}

def generate_mermaid(cfg):
    lines = ["flowchart TD"]
    for n in cfg:
        label = n.label.replace('"', "'")
        lines.append(f'  N{n.id}["{label}"]')
    for n in cfg:
        for e in n.edges:
            lines.append(f"  N{n.id} --> N{e}")
    return "\n".join(lines)

import shutil
import subprocess
import uuid
from typing import Dict, Any

def run_in_docker(code: str) -> Dict[str, Any]:
    """
    Try to run code inside docker if docker binary is available.
    If docker is missing, return a friendly message rather than raising.
    """
    # Check if docker exists
    if shutil.which("docker") is None:
        return {"error": "docker-not-found", "message": "Docker binary not found on host. Install Docker Desktop to enable sandboxed execution."}

    # otherwise attempt to run inside docker
    container_code = f"/tmp/{uuid.uuid4().hex}.py"
    try:
        with open(container_code, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{container_code}:/tmp/code.py", "python:3.10", "python", "/tmp/code.py"],
            capture_output=True, text=True, timeout=10
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as e:
        return {"error": "docker-run-failed", "message": str(e)}


def generate_explanation(ast_data, cfg_nodes, trace, docker_result):
    """
    Uses OpenAI to generate a textual explanation. Falls back to rule-based text
    if OpenAI is not configured or an error occurs.
    """
    # Construct a concise prompt for the model
    # Keep the prompt short to avoid hitting rate/size limits.
    user_payload = {
        "ast": ast_data if isinstance(ast_data, dict) else {},
        "cfg_nodes": [n.to_dict() for n in cfg_nodes],
        "trace": trace,
        "docker": docker_result
    }

    system_msg = "You are CodeAssistX assistant. Explain the code's purpose, control flow and runtime trace concisely."

    user_msg = f"Analyze the following data and provide a short, clear explanation (what the code does, key control-flow points, and runtime outcome):\n\n{json.dumps(user_payload, indent=2)[:4000]}"

    # If OpenAI key is not set, return fallback
    if not os.environ.get("OPENAI_API_KEY"):
        return "Fallback Explanation:\nAPI key not configured. AST root: {}\nNodes: {}\nTrace: {}".format(
            ast_data.get("_type") if isinstance(ast_data, dict) else "N/A",
            len(cfg_nodes),
            trace.get("logs")
        )

    try:
        # Use the chat completion (client-chat) API. You can also use responses.create()
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" / "gpt-4o" depending on availability and cost
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        # New SDK returns structured response; extract assistant content:
        content = ""
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            content = resp.choices[0].message.content
        else:
            content = str(resp)

        return content

    except Exception as e:
        # on any error, return fallback (and log the exception to server console)
        print("OpenAI call error:", e)
        return "Fallback Explanation (OpenAI error): AST root: {}\nNodes: {}\nTrace: {}".format(
            ast_data.get("_type") if isinstance(ast_data, dict) else "N/A",
            len(cfg_nodes),
            trace.get("logs")
        )


@app.post("/analyze")
def analyze(payload: CodeInput):
    code = payload.code or ""
    try:
        tree = ast.parse(code)
        ast_dict = ast_to_dict(tree)
    except Exception as e:
        ast_dict = {"error": str(e)}

    cfg_gen = SimpleCFGGenerator()
    cfg_nodes = cfg_gen.generate(code)
    trace = safe_exec(code)
    docker_out = run_in_docker(code)
    explanation = generate_explanation(ast_dict, cfg_nodes, trace, docker_out)
    mermaid = generate_mermaid(cfg_nodes)

    return {
        "ast": ast_dict,
        "cfg": [n.to_dict() for n in cfg_nodes],
        "mermaid": mermaid,
        "trace": trace,
        "docker": docker_out,
        "explanation": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("codeassistx_mvp_advanced:app", host="0.0.0.0", port=8000, reload=True)
