"""
OpenAI GPT-3.5-turbo Financial Analysis Agent Module

Mirrors the interface of claude_agent.py exactly so the runner can call either agent identically.
Uses OpenAI function-calling (the parallel to Anthropic tool_use).

This module defines tool-using agent that:
    1. Accepts a financial analysis Task
    2. Uses tools (calculate) to gather and process data
    3. Returns a structured response with reasoning trace
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import json
import time
import re
from typing import Any
import openai
from agents.shared import SYSTEM_PROMPT, handle_tool_call

CLIENT = openai.OpenAI()  # Reads OPENAI_API_KEY from environment

# Define the tools available for the agent to call. Here we have a calculate function defines which will
# encourage the agent to not do mathematical computation implicitly, instead it will be forced to think about
# the expression to be computed first and then it will provide that to this tool which will compute the
# value exactly using Python's eval() function. This helps reduce calculation errors.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform a financial calculation. Safely evaluates a mathematical expression. "
                "Use Python-style math: **, /, *, +, -. Supports round(), abs(), pow()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '(211915/168088)**0.5 - 1'"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this calculation computes"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def run_agent(model_name: str, task_prompt: str, max_turns: int = 10) -> dict:
    """
    Run the OpenAI GPT agent on a single task. Returns a result dict with:
        - answer: the final text answer
        - tool_calls: a list of tool calls made
        - turns: number of agentic turns
        - latency_ms: wall-clock time
        - raw_messages: full message trace
        - error: None or exception string

    :param model_name: The name of the model to run e.g. gpt-4.5.
    :param task_prompt: An input str describing the task to be performed.
    :param max_turns: Caps how many times the agent can go back and forth with the API before we force
        it to stop to prevent infinite looping or ultra long task runtimes.
    """
    start = time.time()  # Record how long the entire task run takes
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    tool_calls = []
    turns = 0
    error = None
    final_answer = None

    try:
        for _ in range(max_turns):
            turns += 1
            response = CLIENT.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=2048,
            )

            msg = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls})

            if msg.tool_calls:
                tool_results = []
                for tool_call in msg.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    result = handle_tool_call(tool_call.function.name, arguments)
                    tool_calls.append({
                        "name": tool_call.function.name,
                        "input": arguments,
                        "result_preview": str(result)[:200],
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
                messages.extend(tool_results)  # append all at once

            else:
                # stop or end_turn
                final_answer = msg.content
                break

    except Exception as e:
        error = str(e)

    # Record the total runtime of the agent to complete this task
    latency_ms = int((time.time() - start) * 1000)

    return {
        "answer": final_answer,
        "tool_calls": tool_calls,
        "turns": turns,
        "latency_ms": latency_ms,
        "error": error,
        "message_count": len(messages),
    }
