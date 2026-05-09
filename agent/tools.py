"""
Tool definitions available to the agent.
Each tool has a Python callable + a JSON schema description
that gets passed to the model as context.
"""

import json
import wikipedia as wiki_api


def wikipedia_search(query: str, sentences: int = 3) -> str:
    try:
        return wiki_api.summary(query, sentences=sentences, auto_suggest=False)
    except wiki_api.exceptions.DisambiguationError as e:
        # Try the first suggestion
        return wiki_api.summary(e.options[0], sentences=sentences)
    except wiki_api.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"
    except Exception as e:
        return f"Error: {str(e)}"


def calculator(expression: str) -> str:
    allowed = {k: v for k, v in __builtins__.items()
               if k in ("abs", "round", "min", "max", "sum", "pow")} \
              if isinstance(__builtins__, dict) else {}
    try:
        result = eval(expression, {"__builtins__": allowed})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# Schema used both for model prompt context and for eval validation
TOOL_SCHEMAS = [
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia and return a plain-text summary of the topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or entity to look up."
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of summary sentences to return (default 3)."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '(3 + 5) * 2'."
                }
            },
            "required": ["expression"]
        }
    }
]

TOOL_REGISTRY = {
    "wikipedia_search": wikipedia_search,
    "calculator": calculator,
}


def execute(tool_call: dict) -> str:
    """Execute a parsed tool call dict: {"name": ..., "arguments": {...}}"""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}"
    return TOOL_REGISTRY[name](**args)


def tools_prompt_block() -> str:
    return json.dumps(TOOL_SCHEMAS, indent=2)
