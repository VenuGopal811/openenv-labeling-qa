"""
Episode-level skill score.

Why this exists: the per-step reward pays +1.0 for accepting a clean label,
and 70 to 85 percent of labels are clean. A policy that approves every label
without reading it therefore scores well on raw cumulative reward. Before this
module, that do-nothing policy beat the published llama-3.1-8b-instant
baseline on all three tasks.

The skill score measures an agent against two reference policies run on the
exact same examples it saw:

    skill = (agent - accept_all) / (oracle - accept_all)

    accept_all  approves every label with confidence 0.5 (the strongest
                trivial policy: it avoids the overconfidence penalty)
    oracle      takes the best-scoring action for every example

skill = 0 means no better than approving everything, 1 means perfect, and a
negative value means worse than approving everything. When an episode has no
noisy labels, oracle equals accept_all and skill is undefined (None). Across
many episodes, aggregate by summing numerators and denominators
(see aggregate_skill) rather than averaging per-episode ratios.
"""

from environment.grader import grade

ACCEPT_ALL_CONFIDENCE = 0.5


def accept_all_action(example: dict) -> dict:
    return {"verdict": "correct", "proposed_label": None, "confidence": ACCEPT_ALL_CONFIDENCE}


def _candidate_actions(example: dict) -> list[dict]:
    gold = str(example["gold_label"])
    return [
        {"verdict": "correct", "proposed_label": None, "confidence": ACCEPT_ALL_CONFIDENCE},
        {"verdict": "wrong", "proposed_label": gold, "confidence": ACCEPT_ALL_CONFIDENCE},
        {"verdict": "ambiguous", "proposed_label": None, "confidence": ACCEPT_ALL_CONFIDENCE},
    ]


def accept_all_score(task: str, examples: list[dict]) -> float:
    return sum(grade(task, accept_all_action(ex), ex)["score"] for ex in examples)


def oracle_score(task: str, examples: list[dict]) -> float:
    return sum(max(grade(task, a, ex)["score"] for a in _candidate_actions(ex)) for ex in examples)


def skill(agent: float, accept_all: float, oracle: float) -> float | None:
    headroom = oracle - accept_all
    if headroom <= 0:
        return None
    return (agent - accept_all) / headroom


def aggregate_skill(episodes: list[tuple[float, float, float]]) -> float | None:
    """episodes: (agent, accept_all, oracle) per episode."""
    gain = sum(a - b for a, b, _ in episodes)
    headroom = sum(o - b for _, b, o in episodes)
    if headroom <= 0:
        return None
    return gain / headroom
