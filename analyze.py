"""
Statistical analysis of eval results, with per-agent breakdown.

Usage:
  python analyze.py                        # analyzes most recent run
  python analyze.py --run results/run_...  # specific run file
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Tuple
import numpy as np
import scipy.stats as stats

RESULTS_DIR = Path(__file__).parent / "results"


### Summary Statistics ###

def mean(x) -> float:
    """Computes sample mean"""
    return np.array(x).mean() if x else 0.0


def std(x) -> float:
    """Computes sample standard deviation"""
    return np.array(x).std(ddof=1) if len(x) > 2 else 0.0


def wilson_ci(k: float, n: int, cl: float = 0.95) -> Tuple[float]:
    """
    Computes a Wilson confidence interval lower and upper bound.

    This is a highly accurate method for calculating the confidence interval of a population proportion,
    even with small sample sizes or extreme proportions close to 0 or 1. Unlike the standard Wald interval,
    it never produces impossible bounds less than zero or greater than one.

    :param k: The number of successes out of n trials.
    :param n: The number of total trials.
    :param cl: The desired confidence level e.g. 0.95 for 95%.
    :return: A length-2 tuple containing the lower and upper bound of the CI.
    """
    assert k <= n, "Number of successes (k) cannot exceed total trials (n)"
    if n == 0:
        return (0.0, 1.0)

    alpha = 1 - cl  # Calculate alpha (total error probability shared by both tails)
    z = stats.norm.ppf(1 - alpha / 2)  # Get the Z-score cutoff (Normal Distribution)
    p = k / n  # Compute the success rate
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def t_ci(x, cl: float = 0.95) -> Tuple[float]:
    """
    Returns a confidence interval for the mean of x using a student-t distribution.

    :param x: The input data.
    :param cl: The desired confidence level e.g. 0.95 for 95%.
    :return: A length-2 tuple containing the lower and upper bound of the CI.
    """
    if len(x) < 2:
        return (mean(x), mean(x))

    degrees_of_freedom = len(x) - 1  # Compute the degrees of freedom for a 2-sided CI
    alpha = 1 - cl  # Calculate alpha (total error probability shared by both tails)
    t = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)  # The t-critical value cutoff
    mu = mean(x)
    sigma = std(x)
    half = t * sigma / math.sqrt(len(x))
    return (mu - half, mu + half)


### Load Results ###

def get_latest_summary_filename() -> Path:
    """Returns the file name of the latest run based on the file names."""
    files = sorted(f for f in RESULTS_DIR.glob("run_*.json") if "_summary" not in f.name)
    if not files:
        print("No result files found in results/. Run runner.py first.")
        sys.exit(1)
    return files[-1]


def load_run(path: Path) -> dict:
    """Loads in a JSON file from disk and returns the data as a dict"""
    with open(path) as f:
        return json.load(f)


### Per-Agent Task Stats ###

def compute_task_stats(trials: list, task_id: str, task_name: str, tier: int, grader_type: str) -> dict:
    """
    Computs a summary of how well an agent performed based on a list of trial results.

    :param trials: A list of trial results (dictionaries) summarizing the agent's performance on a given task.
    :param task_id: The id of the task for record keeping.
    :param task_name: The name of the task for record keeping.
    :param tier: The difficulty of the task for record keeping.
    :param grader_type: The type of grading applied to the task for record keeping.
    :return: A dictionary summary of the performance results across all trials of a given task.
    """
    scores = [t["score"] for t in trials]  # Extract scores
    passes = [t["passed"] for t in trials]  # Extract how many trials passed
    n = len(trials)  # The total number of trials run
    k = sum(passes)  # The total number of successful trial runs achieved by the agent
    return {
        "task_id": task_id,
        "task_name": task_name,
        "tier": tier,
        "grader_type": grader_type,
        "n_trials": n,
        "pass_rate": k / n if n else 0,  # The fraction of trials that the agent was successful on
        "pass_rate_ci_95": wilson_ci(k, n),  # 95% confidence interval of the pass rate
        "score_mean": mean(scores),
        "score_std": std(scores),
        "score_ci_95": t_ci(scores, n),  # Assuming a t-distribution, this computes the 95% CI of the mean
        "avg_turns": mean([t["turns"] for t in trials]),  # Average number of turns required to compute
        "avg_latency_ms": mean([t["latency_ms"] for t in trials]),  # Avg runtime by the agent
        "n_tool_calls": mean([len(t.get("tool_calls", [])) for t in trials]),
        "n_agent_errors": sum(1 for t in trials if t.get("agent_error")),
        "failures": [
            {
                "trial": t["trial"],
                "score": t["score"],
                "answer_preview": (t.get("answer") or "")[:200],
                "explanation": t.get("grade_explanation", ""),
            }
            for t in trials if not t["passed"]
        ],
    }


def analyze(results: list) -> dict:
    """
    Analyzes a list of results for each agent across many tasks with potentially many trials each.
    """
    # Detect agents present
    agents = list(dict.fromkeys(r.get("agent", "claude") for r in results))  # preserve order, dedupe
    agent_labels = {r.get("agent", "claude"): r.get("agent_label", r.get("agent", "claude")) for r in results}

    # Per-agent analysis
    agent_stats = {}
    for agent_key in agents:
        agent_results = [r for r in results if r.get("agent", "claude") == agent_key]
        by_task = defaultdict(list)
        for r in agent_results:
            by_task[r["task_id"]].append(r)

        task_stats = {}
        for task_id, trials in sorted(by_task.items()):
            t0 = trials[0]
            task_stats[task_id] = compute_task_stats(
                trials, task_id, t0["task_name"], t0["tier"], t0["grader_type"]
            )

        # Tier aggregates for this agent
        tier_stats = defaultdict(lambda: {"scores": [], "passes": [], "tasks": []})
        for ts in task_stats.values():
            for r in agent_results:
                if r["task_id"] == ts["task_id"]:
                    tier_stats[ts["tier"]]["scores"].append(r["score"])
                    tier_stats[ts["tier"]]["passes"].append(r["passed"])
            if ts["task_id"] not in tier_stats[ts["tier"]]["tasks"]:
                tier_stats[ts["tier"]]["tasks"].append(ts["task_id"])

        tier_summary = {}
        for tier, data in sorted(tier_stats.items()):
            sc, pa = data["scores"], data["passes"]
            tier_summary[tier] = {
                "tier": tier,
                "tier_label": {1: "Easy", 2: "Medium", 3: "Hard"}.get(tier, str(tier)),
                "tasks": data["tasks"],
                "n_trials": len(sc),
                "pass_rate": sum(pa) / len(pa) if pa else 0,
                "score_mean": mean(sc),
                "score_std": std(sc),
            }

        all_scores = [r["score"] for r in agent_results]
        all_passes = [r["passed"] for r in agent_results]
        n_total = len(all_scores)
        k_total = sum(all_passes)

        agent_stats[agent_key] = {
            "agent": agent_key,
            "agent_label": agent_labels[agent_key],
            "total_trials": n_total,
            "overall_pass_rate": k_total / n_total if n_total else 0,
            "overall_pass_ci_95": wilson_ci(k_total, n_total),
            "overall_score_mean": mean(all_scores),
            "overall_score_std": std(all_scores),
            "task_stats": task_stats,
            "tier_summary": tier_summary,
            "failure_modes": _classify_failures(agent_results),
        }

    # Head-to-head comparison
    head_to_head = []
    if len(agents) >= 2:
        all_task_ids = sorted(set(r["task_id"] for r in results))
        for task_id in all_task_ids:
            row = {"task_id": task_id}
            t0 = next(r for r in results if r["task_id"] == task_id)
            row["task_name"] = t0["task_name"]
            row["tier"] = t0["tier"]
            row["grader_type"] = t0["grader_type"]
            for agent_key in agents:
                agent_trials = [r for r in results
                                if (r["task_id"] == task_id) and (r.get("agent", "claude") == agent_key)]
                if agent_trials:
                    scores = [t["score"] for t in agent_trials]
                    passes = [t["passed"] for t in agent_trials]
                    row[f"{agent_key}_pass_rate"] = sum(passes) / len(passes)
                    row[f"{agent_key}_score_mean"] = mean(scores)
                    row[f"{agent_key}_score_std"] = std(scores)
                else:
                    row[f"{agent_key}_pass_rate"] = None
                    row[f"{agent_key}_score_mean"] = None
                    row[f"{agent_key}_score_std"] = None

            # Winner per task (by score mean)
            scores_by_agent = {
                a: row.get(f"{a}_score_mean") for a in agents
                if row.get(f"{a}_score_mean") is not None
            }
            if scores_by_agent:
                winner = max(scores_by_agent, key=scores_by_agent.get)
                margin = max(scores_by_agent.values()) - min(scores_by_agent.values())
                row["winner"] = winner if margin > 0.01 else "tie"
                row["margin"] = round(margin, 4)
            head_to_head.append(row)

    return {
        "agents": agents,
        "agent_labels": agent_labels,
        "agent_stats": agent_stats,
        "head_to_head": head_to_head,
    }


def _classify_failures(results: list) -> list:
    """
    Helper function that classifies the types of failures (if any) in the results.
    """
    categories = defaultdict(list)
    for r in results:
        if r.get("passed"):
            continue
        exp = (r.get("grade_explanation") or "").lower()
        ans = (r.get("answer") or "").lower()
        if r.get("agent_error"):
            cat = "Agent/API Error"
        elif "could not extract" in exp or "no number" in exp:
            cat = "Number Extraction Failure"
        elif "outside range" in exp or ("error" in exp and "%" in exp):
            cat = "Calculation Error"
        elif "not found in" in exp:
            cat = "Answer Format Mismatch"
        elif len(ans) < 20:
            cat = "Truncated / Empty Answer"
        elif r.get("score", 0) > 0.4:
            cat = "Partial Credit - Rubric Gaps"
        else:
            cat = "Content Error"
        categories[cat].append({
            "task_id": r["task_id"],
            "trial": r["trial"],
            "score": r.get("score", 0),
            "explanation": r.get("grade_explanation", "")[:200],
        })
    return [
        {"category": cat, "count": len(inst), "examples": inst[:3]}
        for cat, inst in sorted(categories.items(), key=lambda x: -len(x[1]))
    ]


def print_report(summary: dict):
    """Helper function for printing a summary report to the console after running."""
    agents = summary["agents"]
    agent_labels = summary["agent_labels"]
    agent_stats = summary["agent_stats"]
    h2h = summary["head_to_head"]

    print("\n" + "=" * 70)
    print("  FinEval - Statistical Analysis Report")
    print("=" * 70)

    for agent_key in agents:
        s = agent_stats[agent_key]
        ci = s["overall_pass_ci_95"]
        print(f"\n  [{s['agent_label']}]")
        print(f"    Pass rate : {s['overall_pass_rate'] * 100:.1f}%  "
              f"(95% CI: {ci[0] * 100:.1f}-{ci[1] * 100:.1f}%)")
        print(f"    Mean score: {s['overall_score_mean']:.3f} ± {s['overall_score_std']:.3f}")
        print(f"    Trials    : {s['total_trials']}")

    if len(agents) >= 2 and h2h:
        print("\n" + "─" * 70)
        print("  Head-to-Head Comparison")
        print("─" * 70)
        col_w = 14
        header = f"  {'ID':<5} {'Task':<30}"
        for a in agents:
            header += f"  {agent_labels[a][:col_w]:<{col_w}}"
        header += "  Winner"
        print(header)
        print("  " + "─" * 66)
        for row in h2h:
            line = f"  {row['task_id']:<5} {row['task_name'][:29]:<30}"
            for a in agents:
                pr = row.get(f"{a}_pass_rate")
                sm = row.get(f"{a}_score_mean")
                cell = f"{pr * 100:.0f}% / {sm:.2f}" if pr is not None else "  n/a  "
                line += f"  {cell:<{col_w}}"
            winner_key = row.get("winner", "")
            winner_label = agent_labels.get(winner_key, winner_key)
            margin = row.get("margin", 0)
            line += f"  {winner_label} (+{margin:.2f})" if winner_key != "tie" else "  tie"
            print(line)

    for agent_key in agents:
        s = agent_stats[agent_key]
        if s["failure_modes"]:
            print(f"\n  Failure Modes — {s['agent_label']}")
            print("  " + "-" * 40)
            for fm in s["failure_modes"]:
                print(f"  [{fm['count']:2d}x] {fm['category']}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    path = Path(args.run) if args.run else get_latest_summary_filename()
    print(f"  Analyzing: {path.name}")
    results = load_run(path)
    summary = analyze(results)
    print_report(summary)

    out = RESULTS_DIR / f"{path.stem}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved → {out}")
    return summary


if __name__ == "__main__":
    main()
