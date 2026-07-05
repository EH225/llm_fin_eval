"""
This module contains code shared by all agents.
"""
import re

# Define the system prompt to feed to the agent as pretext before any query. These instructions help
# guide the agent's behavior towards the desired outcome via prompt engineering.
SYSTEM_PROMPT = """You are a rigorous financial analyst AI. You have access to tools to perform calculations.

When given a financial analysis task:
1. Use calculate for any arithmetic to avoid errors
2. Be precise with numbers — cite exact figures from the filing
3. Structure your final answer clearly
4. If asked for a specific format (word count, numbered list, percentage format), follow it exactly

Always show your calculation steps explicitly."""


def _safe_eval(expr: str) -> float:
    """
    Safely evaluates a mathematical expression e.g. (10 + 5) / 12
    """
    allowed = re.compile(r'^[\d\s\+\-\*\/\.\(\)\*\*e\_]+$')  # Only allowed characters
    cleaned = expr.strip()  # Remove extra white space
    if not allowed.match(cleaned):
        return "Calculation error: expression contains disallowed characters"
    try:
        result = eval(cleaned, {"__builtins__": {}, "round": round, "abs": abs, "pow": pow})
        return float(result)
    except Exception as e:
        return f"Calculation error: {e}"


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Handles tool calls by an agent. For calculate, evaluate the expression safely.
    """
    if tool_name == "calculate":
        expr = tool_input.get("expression", "")
        result = _safe_eval(expr)
        desc = tool_input.get("description", "")
        return f"Result of '{expr}' ({desc}): {result}"
    else:
        return f"Unknown tool: {tool_name}"
