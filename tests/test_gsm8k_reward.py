import inspect

import pytest

import grpo.rewards.gsm8k as gsm8k_module
from grpo.rewards.gsm8k import (
    PROMPT_SUFFIX,
    gsm8k_reward,
    gsm8k_reward_with_format,
)
from grpo.rewards.verl_gsm8k import compute_score as verl_compute_score

LONG = "step " * 80 + "\nso the result follows.\n#### 72"  # > 300 chars
assert len(LONG) > 300

CASES = [
    # (test id, completion, ground_truth, expected reward)
    ("plain_integer", "The answer is\n#### 72", "72", 1.0),
    ("comma_stripped", "#### 1,234", "1234", 1.0),
    # "$" is not in the strict regex's character class, so extraction fails
    # outright; verl's .replace("$","") is unreachable in strict mode. The
    # wrapper inherits this on purpose -- PROMPT_SUFFIX forbids symbols.
    ("dollar_sign_fails_strict", "#### $18", "18", 0.0),
    # The one wrapper deviation: trailing "." stripped before comparison.
    ("trailing_dot", "I conclude.\n#### 72.", "72", 1.0),
    ("multiple_markers_last_wins", "#### 5 is wrong, actually\n#### 7", "7", 1.0),
    # Correct number present but no #### marker: strict gives 0.0. Flexible
    # mode would score this 1.0, so this row is the behavioral proof that
    # flexible is not reachable. No fallback parsing, ever.
    ("no_marker_no_fallback", "After careful thought, the answer is 72.", "72", 0.0),
    ("wrong_answer_well_formatted", "#### 71", "72", 0.0),
    ("negative_number", "#### -5", "-5", 1.0),
    ("empty_string", "", "72", 0.0),
    ("garbage", "@@@@ ???? ####x 12abc ####", "72", 0.0),
    # > 300 chars with the answer at the end: the vendored clip keeps the
    # tail, so this must still score.
    ("long_completion_clip", LONG, "72", 1.0),
    # No space after the hashes: the strict regex requires "#### ", so no match.
    ("no_space_after_hashes", "The answer is ####72", "72", 0.0),
    ("text_after_marker", "Adding them up gives 42. #### 42\n\nLet me double-check: yes, confident.", "42", 1.0),
    # Needs the wrapper's numeric-equivalence deviation: string compare alone
    # would score "18.0" vs "18" as 0.0.
    ("decimal_equivalent", "The total is #### 18.0", "18", 1.0),
]


@pytest.mark.parametrize(
    "completion, ground_truth, expected",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_reward_table(completion, ground_truth, expected):
    assert gsm8k_reward(completion, ground_truth) == expected


def test_trailing_dot_is_a_real_verl_failure():
    # Prove the wrapper's deviation fixes an actual vendored-scorer failure,
    # not a hypothetical one: raw verl strict scores "#### 72." as 0.
    assert verl_compute_score("#### 72.", "72", method="strict") == 0.0
    assert gsm8k_reward("#### 72.", "72") == 1.0


@pytest.mark.parametrize("bad", [None, 72, b"#### 72", ["#### 72"]])
def test_never_raises(bad):
    assert gsm8k_reward(bad, "72") == 0.0


def test_flexible_mode_not_reachable():
    # Static check: the wrapper module never mentions flexible mode at all,
    # so it cannot pass it to the vendored extractor. (The behavioral check
    # is the no_marker_no_fallback table row, which flexible would score 1.0.)
    source = inspect.getsource(gsm8k_module)
    assert "flexible" not in source
    assert 'method="strict"' in source


@pytest.mark.parametrize(
    "completion, ground_truth, expected",
    [
        ("#### 72", "72", (1.0, True)),  # formatted and correct
        ("#### 71", "72", (0.0, True)),  # formatted, wrong answer
        ("the answer is 72", "72", (0.0, False)),  # no marker: format failure
        ("", "72", (0.0, False)),
        (None, "72", (0.0, False)),  # never raises, reports unformatted
    ],
    ids=["correct", "wrong_but_formatted", "unformatted", "empty", "non_string"],
)
def test_reward_with_format_tuple(completion, ground_truth, expected):
    assert gsm8k_reward_with_format(completion, ground_truth) == expected


def test_reward_delegates_to_with_format():
    # One extraction path: the scalar reward must always equal tuple[0].
    for completion in ["#### 72", "#### 71", "no marker 72", "", "#### 72."]:
        reward, _ = gsm8k_reward_with_format(completion, "72")
        assert gsm8k_reward(completion, "72") == reward


def test_prompt_suffix_contract():
    # The suffix must literally show the marker format it demands.
    assert "#### <answer>" in PROMPT_SUFFIX
    # And a completion that follows the suffix's own example must score.
    assert gsm8k_reward("blah blah\n#### 72", "72") == 1.0
