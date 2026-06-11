"""Thin wrapper around the vendored verl GSM8K scorer.

All deviations from verl behavior live HERE, never in verl_gsm8k.py.
Current deviations, in full:
  1. A trailing "." is stripped from the extracted answer before comparison
     (the verl strict regex captures "72." for "#### 72." and then fails the
     exact-match against "72").
  2. Numeric equivalence: if the exact string match fails, the answer matches
     when float(answer) == float(ground_truth), so "18.0" matches "18".
     verl uses string equality only.
  3. gsm8k_reward never raises; any exception scores 0.0.

Known strict-mode behavior we inherit on purpose (not deviations): the strict
regex is `#### (\\-?[0-9\\.\\,]+)`, so a "$" or any unit after "#### " makes
extraction fail entirely -> 0.0. PROMPT_SUFFIX below forbids symbols/units so
the format contract matches what the scorer accepts.
"""

from grpo.rewards.verl_gsm8k import extract_solution

# Appended verbatim to every GSM8K prompt. This is the format contract the
# strict scorer enforces; keep the two in sync.
PROMPT_SUFFIX = (
    "\nThink step by step, then write your final answer on its own last line"
    ' in exactly this form:\n#### <answer>\nwhere <answer> is a plain number'
    " with no commas, units, or symbols (for example: #### 72)."
)


def gsm8k_reward(completion: str, ground_truth: str) -> float:
    try:
        answer = extract_solution(completion, method="strict")
        if answer is None:
            return 0.0
        if answer.endswith("."):
            answer = answer[:-1]
        if answer == ground_truth:
            return 1.0
        try:
            return 1.0 if float(answer) == float(ground_truth) else 0.0
        except (TypeError, ValueError):
            return 0.0
    except Exception:
        return 0.0
