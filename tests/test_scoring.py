"""Grader and skill-score tests. Synthetic examples only, no dataset download."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.grader import grade  # noqa: E402
from environment.scoring import (  # noqa: E402
    accept_all_action,
    accept_all_score,
    aggregate_skill,
    oracle_score,
    skill,
)

GOLD = {"easy": "1", "medium": "contradiction", "hard": "5"}
WRONG = {"easy": "0", "medium": "neutral", "hard": "6"}
TASKS = list(GOLD)


def make_examples(task, n_clean, n_noisy):
    clean = [{"gold_label": GOLD[task], "ai_label": GOLD[task], "is_noisy": False}] * n_clean
    noisy = [{"gold_label": GOLD[task], "ai_label": WRONG[task], "is_noisy": True}] * n_noisy
    return clean + noisy


def flag(task, label):
    return {"verdict": "wrong", "proposed_label": label, "confidence": 0.9}


@pytest.mark.parametrize("task", TASKS)
def test_accept_all_has_zero_skill(task):
    examples = make_examples(task, 8, 2)
    agent = sum(grade(task, accept_all_action(ex), ex)["score"] for ex in examples)
    assert skill(agent, accept_all_score(task, examples), oracle_score(task, examples)) == 0.0


@pytest.mark.parametrize("task", TASKS)
def test_perfect_agent_has_skill_one(task):
    examples = make_examples(task, 8, 2)
    agent = sum(
        grade(task, flag(task, GOLD[task]) if ex["is_noisy"] else accept_all_action(ex), ex)["score"]
        for ex in examples
    )
    assert skill(agent, accept_all_score(task, examples), oracle_score(task, examples)) == pytest.approx(1.0)


@pytest.mark.parametrize("task", TASKS)
def test_flagging_everything_has_negative_skill(task):
    examples = make_examples(task, 8, 2)
    agent = sum(grade(task, flag(task, WRONG[task]), ex)["score"] for ex in examples)
    assert skill(agent, accept_all_score(task, examples), oracle_score(task, examples)) < 0


@pytest.mark.parametrize("task", TASKS)
def test_proposing_the_gold_label_earns_more_than_flagging_alone(task):
    # Regression: the bonus used to be capped away by min(1.0, 1.0 + 0.5).
    noisy = make_examples(task, 0, 1)[0]
    with_fix = grade(task, flag(task, GOLD[task]), noisy)["score"]
    flag_only = grade(task, flag(task, "not-a-label"), noisy)["score"]
    assert with_fix > flag_only


@pytest.mark.parametrize("task", TASKS)
def test_rewards_stay_in_declared_range(task):
    actions = [
        accept_all_action({}),
        {"verdict": "correct", "proposed_label": None, "confidence": 1.0},
        {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5},
        flag(task, GOLD[task]),
        flag(task, WRONG[task]),
    ]
    for ex in make_examples(task, 1, 1):
        for action in actions:
            assert -1.0 <= grade(task, action, ex)["score"] <= 1.0


def test_skill_is_undefined_without_noisy_examples():
    examples = make_examples("easy", 10, 0)
    base = accept_all_score("easy", examples)
    assert skill(base, base, oracle_score("easy", examples)) is None


def test_aggregate_skill_pools_episodes():
    # Episode with no headroom contributes nothing; the other decides the result.
    assert aggregate_skill([(10.0, 10.0, 10.0), (8.0, 7.0, 9.0)]) == pytest.approx(0.5)
