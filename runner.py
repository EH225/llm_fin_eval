"""
runner.py - Run repeated evaluation trials across multiple agents to compare results.

Usage:
  python runner.py                           # 3 trials, all tasks, both agents
  python runner.py --trials 5                # 5 trials, all tasks, both agents
  python runner.py --tasks T1,T3             # Run only specific tasks ## TODO FIX THIS IT DOESN"T WORK"
  python runner.py --agents openai           # Run the openai only
  python runner.py --agents claude           # Run the claude agent only
  python runner.py --agents claude, openai   # Run both agents (default)
  python runner.py --dry-run                 # Validate setup without API calls for testing
"""

import argparse
import json
import sys
import time
import uuid
import os
from datetime import datetime, timezone
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks.task_registry import TASKS, TASK_MAP
from graders.grader import grade_response

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define what models to use for each type of agent
AGENT_MODELS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-5.4",
}


########################
### Helper Functions ###
########################

def load_agent(agent_key: str) -> Callable:
    """
    Loads the agent specified by returning the callable function associated with that agent to
    execute tasks with.

    :param agent_key: The name of the agent to laod i.e. openai or claude.
    :returns: A callable function run_agent that can be used to run query commands.
    """
    if agent_key == "claude":
        from agents.claude_agent import run_agent
        return run_agent
    elif agent_key == "openai":
        from agents.openai_agent import run_agent
        return run_agent
    else:
        raise ValueError(f"Unknown agent: {agent_key}. Choose from: claude, openai")


def run_trial(task, trial_num: int, agent_class: str, run_agent_fn: Callable, dry_run: bool = False) -> dict:
    """
    Runs a single agent trial and grades it.

    :param task: A Task object containing all the info required to run and grade a given task.
    :param trial_num: An integer denoting which trial iteration this function call is for a given task.
    :param agent_class: The name of the agent to laod i.e. openai or claude.
    :param run_agent_fn: A callable function that will run the task with the agent.
    :param dry_run: A bool indicating if this is to be run without API calls for debug testing.
    :returns: A dictionary summary of the trail run which includes details about the agent, task, and eval
        score obtained.
    """
    model_name = AGENT_MODELS.get(agent_class, agent_class)
    print(f"    [{model_name}] Trial {trial_num}... ", end="", flush=True)

    if dry_run:  # Runs the setup without API calls for debug testing
        agent_result = {
            "answer": f"[DRY RUN] Mock answer for {task.id} trial {trial_num} ({model_name})",
            "tool_calls": [{"name": "retrieve_filing", "input": {}, "result_preview": "..."}],
            "turns": 2,
            "latency_ms": 100,
            "error": None,
            "message_count": 4,
        }

    else:  # Run with the actual API calls to generate results
        agent_result = run_agent_fn(model_name, task.prompt + "\n" + task.context)

    # Grade the agent on the task it was asked to complete, generate eval scoring
    grade = grade_response(task, agent_result)

    # Construct the output return summary as a dict, include details on the agent, the task and
    # the evaluation score obtained by the agent for this trial run
    result = {
        "run_id": str(uuid.uuid4())[:8],
        "agent": agent_class,
        "agent_label": model_name,
        "task_id": task.id,
        "task_name": task.name,
        "tier": task.tier,
        "trial": trial_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer": agent_result.get("answer", ""),
        "tool_calls": agent_result.get("tool_calls", []),
        "turns": agent_result.get("turns", 0),
        "latency_ms": agent_result.get("latency_ms", 0),
        "agent_error": agent_result.get("error"),
        "score": grade.score,
        "passed": grade.passed,
        "raw_score": str(grade.raw_score)[:500] if grade.raw_score else None,
        "grade_explanation": grade.explanation,
        "grader_type": grade.grader_type,
    }

    status = "✓ PASS" if grade.passed else "X - FAIL"
    print(f"{status} (score={grade.score:.2f}, {agent_result.get('latency_ms', 0)}ms)")
    return result


def save_results(results: list, run_name: str) -> str:
    """
    Saves a list of results (a list of dictionaries, one for each trial run) to disk as a JSON.

    :param results: A list of result dictionaries (one for each trial run).
    :param run_name: The name of the run which is used to give the file a unique name.
    :returns: The file path to which the results were saved.
    """
    path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {len(results)} results â†’ {path}")
    return path


def main():
    # Parse input args from the command line
    parser = argparse.ArgumentParser(description="FinEval trial runner")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per task per agent")
    parser.add_argument("--tasks", default="all", help="Comma-separated task IDs or 'all'")
    parser.add_argument("--agents", default="claude,openai", help="Comma-separated agent keys")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use fake results")
    args = parser.parse_args()

    # Select tasks to be run
    if args.tasks == "all":
        selected_tasks = TASKS
    else:
        ids = [t.strip() for t in args.tasks.split(",")]
        selected_tasks = [TASK_MAP[i] for i in ids if i in TASK_MAP]
        if not selected_tasks:
            print(f"ERROR: No valid task IDs. Valid: {list(TASK_MAP.keys())}")
            sys.exit(1)

    # Select agents
    selected_agents = [a.strip().lower() for a in args.agents.split(",")]
    invalid = [a for a in selected_agents if a not in AGENT_MODELS]
    if invalid:
        print(f"ERROR: Unknown agents: {invalid}. Valid: {list(AGENT_MODELS.keys())}")
        sys.exit(1)

    # Load agent functions (catches missing packages early)
    agent_fns = {}
    for key in selected_agents:
        try:
            agent_fns[key] = load_agent(key)
        except Exception as e:
            print(f"ERROR loading agent '{key}': {e}")
            sys.exit(1)

    # Generate a unique name for this run using the current time and date for the output filename
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    total_trials = len(selected_tasks) * args.trials * len(selected_agents)  # All trials to be run

    # Print a quick high-level summary of what will be run for the user before starting execution
    print(f"\n{'=' * 65}")
    print(f"  FinEval Run: {run_name}")
    print(f"  Tasks:   {[t.id for t in selected_tasks]}")
    print(f"  Agents:  {[AGENT_MODELS[a] for a in selected_agents]}")
    print(f"  Trials per task per agent: {args.trials}")
    print(f"  Total trials: {total_trials}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'=' * 65}\n")

    # Run each trial for each agent and aggregate the results across all API calls
    all_results = []  # Aggregate all results (dicts) into a list
    for task in selected_tasks:
        print(f"\n[{task.id}] {task.name} (Tier {task.tier}, grader={task.grader_type})")
        for agent_class in selected_agents:
            for trial in range(1, args.trials + 1):
                result = run_trial(task, trial, agent_class, agent_fns[agent_class], dry_run=args.dry_run)
                all_results.append(result)
                if not args.dry_run:
                    time.sleep(1)

    # Cache the results to disk as a JSON file after all tasks have completed
    save_path = save_results(all_results, run_name)

    # Print a per agent summary of the results
    print(f"\n{'=' * 65}")
    for agent_key in selected_agents:
        agent_results = [r for r in all_results if r["agent"] == agent_key]
        passed = sum(1 for r in agent_results if r["passed"])
        n = len(agent_results)
        label = AGENT_MODELS[agent_key]
        print(f"  {label}: {passed}/{n} passed ({passed / n * 100:.1f}%)" if n else f"  {label}: no results")
    print("\n  Run analyze.py to see variance + failure analysis")
    print("  Run dashboard.py to generate the HTML report")
    print(f"{'=' * 65}\n")
    return str(save_path)


if __name__ == "__main__":
    main()
