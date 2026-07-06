"""
Graders for the FinEval suite.

Four grader types:
  1. ExactGrader       — string normalization + alias matching
  2. ToleranceGrader   — numeric extraction + % tolerance check
  3. RangeGrader       — numeric extraction + [min, max] check
  4. LLMRubricGrader   — calls Claude to score against a weighted rubric
"""

import re
import json
from dataclasses import dataclass
from typing import Any
import anthropic

client = anthropic.Anthropic() # This is used as an automatic grader for the written responses


@dataclass
class GradeResult:
    score: float  # 0.0–1.0
    passed: bool
    raw_score: Any  # type-specific detail
    explanation: str
    grader_type: str


### Helper Functions ###

def _extract_number(text: str, ground_truth: float = None) -> float | None:
    """
    Extracts the most relevant number from text as the answer from the agent to be graded.

    Strategy:
      1. Look for a number near answer-signal words (ratio, CAGR, is, was, =)
      2. If ground_truth provided, pick the candidate closest to it
      3. Fall back to last number in text (answers tend to conclude with the result)
    """
    # Strip thousands-separator commas but keep decimal points
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    cleaned = re.sub(r'[$%]', '', cleaned)

    # Find all numbers with positions
    matches = list(re.finditer(r'-?\d+(?:\.\d+)?', cleaned))
    if not matches:
        return None
    candidates = [float(m.group()) for m in matches]

    # 1. Look for a number preceded by signal phrases
    signal_re = re.compile(
        r'(?:ratio|cagr|rate|equals?|result|answer|therefore|thus|gives?|yielding?|calculates? to|is|was)'
        r'\s*:?\s*\$?(-?\d[\d]*(?:\.\d+)?)\s*%?',
        re.IGNORECASE
    )
    signal_hits = signal_re.findall(cleaned)
    if signal_hits:
        try:
            return float(signal_hits[-1].replace(',', ''))
        except ValueError:
            pass

    # 2. If ground_truth known, pick closest candidate
    if ground_truth is not None:
        return min(candidates, key=lambda x: abs(x - ground_truth))
    else:
        # 3. Fall back to last number (conclusions come at the end)
        return candidates[-1]


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower().replace(',', '').replace('$', '').replace('%', ''))


### Grader 1: Exact match ###

class ExactGrader:
    """
    Passes if the answer contains any of the canonical answer or its aliases, after normalization.
    """

    def grade(self, answer: str, ground_truth: Any, params: dict) -> GradeResult:
        if answer is None:
            return GradeResult(0.0, False, None, "No answer returned by agent", "exact")

        norm_answer = _normalize(answer)
        canonical = _normalize(str(ground_truth))
        aliases = [_normalize(a) for a in params.get("aliases", [])]

        targets = [canonical] + aliases
        matched = next((t for t in targets if t in norm_answer), None)

        if matched:
            return GradeResult(1.0, True, matched, f"Found '{matched}' in answer", "exact")
        else:
            return GradeResult(
                0.0, False, None,
                f"Expected one of {targets!r}, not found in: '{norm_answer[:120]}'",
                "exact"
            )


### Grader 2: Numeric tolerance ###

class ToleranceGrader:
    """
    Extracts a number from the answer and checks that it is within
    tolerance_pct% of the ground truth.
    """

    def grade(self, answer: str, ground_truth: float, params: dict) -> GradeResult:
        if answer is None:
            return GradeResult(0.0, False, None, "No answer returned", "tolerance")

        tol = params.get("tolerance_pct", 1.0)
        unit = params.get("unit", "")
        extracted = _extract_number(answer, ground_truth=ground_truth)

        if extracted is None:
            return GradeResult(
                0.0, False, None,
                f"Could not extract a number from answer: '{answer[:120]}'",
                "tolerance"
            )

        pct_error = abs(extracted - ground_truth) / abs(ground_truth) * 100
        passed = pct_error <= tol
        score = max(0.0, 1.0 - pct_error / (tol * 4))  # partial credit
        score = min(1.0, score)
        if passed:
            score = 1.0

        return GradeResult(
            score, passed, extracted,
            f"Got {extracted} {unit}, expected {ground_truth} {unit} "
            f"(error {pct_error:.2f}%, tolerance {tol}%)",
            "tolerance"
        )


### Grader 3: Range check ###

class RangeGrader:
    """
    Passes if the extracted number falls within [min, max].
    Partial credit for being close.
    """

    def grade(self, answer: str, ground_truth: int, params: dict) -> GradeResult:
        if answer is None:
            return GradeResult(0.0, False, None, "No answer returned", "range")

        lo = params.get("min", ground_truth - 2)
        hi = params.get("max", ground_truth + 2)
        extracted = _extract_number(answer, ground_truth=ground_truth)

        if extracted is None:
            return GradeResult(
                0.0, False, None,
                f"Could not extract a number from: '{answer[:120]}'",
                "range")

        if lo <= extracted <= hi:
            return GradeResult(
                1.0, True, extracted,
                f"Got {extracted}, within range [{lo}, {hi}]",
                "range")

        else:
            dist = min(abs(extracted - lo), abs(extracted - hi))
            score = max(0.0, 1.0 - dist / (hi - lo))
            return GradeResult(
                score, False, extracted,
                f"Got {extracted}, outside range [{lo}, {hi}] (distance {dist:.1f})",
                "range")


### Grader 4: LLM Rubric ###

RUBRIC_SYSTEM = """You are an expert financial analysis examiner.
Your job is to evaluate a student's answer against a detailed rubric.
Be strict but fair. Look for technical accuracy, not just keyword presence.
Respond ONLY with valid JSON, no markdown fences."""

RUBRIC_PROMPT_TEMPLATE = """
## Task
{task_prompt}

## Ground Truth / Key Information
{ground_truth_json}

## Student Answer
{answer}

## Rubric
{rubric_json}

For each rubric criterion, award points on a 0-to-weight scale (can be fractional).
Return a JSON object with this exact structure:
{{
  "scores": [
    {{"criterion": "...", "weight": N, "awarded": N, "reason": "..."}}
  ],
  "total_score": N,
  "max_score": N,
  "passed": true/false,
  "summary": "One sentence overall assessment"
}}
"""


class LLMRubricGrader:
    """
    Uses Claude as a judge, scoring each rubric criterion independently.
    Returns a normalized 0–1 score.
    """

    def grade(self, answer: str, ground_truth: Any, params: dict, task_prompt: str = "") -> GradeResult:
        if answer is None:
            return GradeResult(0.0, False, None, "No answer returned", "llm_rubric")

        rubric = params.get("rubric", [])
        max_score = params.get("max_score", 100)
        pass_threshold = params.get("pass_threshold", 60)

        prompt = RUBRIC_PROMPT_TEMPLATE.format(
            task_prompt=task_prompt,
            ground_truth_json=json.dumps(ground_truth, indent=2),
            answer=answer,
            rubric_json=json.dumps(rubric, indent=2),
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=RUBRIC_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            # Strip any accidental markdown
            raw = re.sub(r'```json|```', '', raw).strip()
            result = json.loads(raw)

            total = result.get("total_score", 0)
            passed = total >= pass_threshold
            score = total / max_score

            return GradeResult(score, passed, result, result.get("summary", ""), "llm_rubric")

        except Exception as e:
            return GradeResult(0.0, False, None, f"LLM grader error: {e}", "llm_rubric")


### Factory ###

def get_grader(grader_type: str):
    return {
        "exact": ExactGrader(),
        "tolerance": ToleranceGrader(),
        "range": RangeGrader(),
        "llm_rubric": LLMRubricGrader(),
    }[grader_type]


def grade_response(task, agent_result: dict) -> GradeResult:
    """Top-level grading entry point."""
    grader = get_grader(task.grader_type)
    answer = agent_result.get("answer", "")

    if task.grader_type == "llm_rubric":
        return grader.grade(answer, task.ground_truth, task.grader_params, task.prompt)
    else:
        return grader.grade(answer, task.ground_truth, task.grader_params)
