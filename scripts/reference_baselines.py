"""
Run reference (non-LLM) policies through the environment and record their scores.

No API key needed. Any agent result should be read against these numbers:
an agent is only useful if it beats accept_all.

    python scripts/reference_baselines.py --episodes 200

Writes results/reference_baselines.json, which is the source for the
reference table in the README.
"""

import argparse
import datetime
import json
import os
import random
import statistics
import subprocess
import sys
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.env import LabelingQAEnv  # noqa: E402
from environment.models import Action  # noqa: E402
from environment.scoring import aggregate_skill  # noqa: E402

TASKS = ["easy", "medium", "hard"]


def accept_all(obs, example, rng):
    return {"verdict": "correct", "proposed_label": None, "confidence": 0.5}


def always_ambiguous(obs, example, rng):
    return {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5}


def flag_all(obs, example, rng):
    others = [label for label in obs.label_options if label != obs.ai_label]
    return {"verdict": "wrong", "proposed_label": rng.choice(others), "confidence": 0.5}


def random_verdict(obs, example, rng):
    verdict = rng.choice(["correct", "wrong", "ambiguous"])
    if verdict == "wrong":
        return flag_all(obs, example, rng)
    return {"verdict": verdict, "proposed_label": None, "confidence": 0.5}


def oracle(obs, example, rng):
    # Reads the hidden gold label. Sanity check only: must score skill 1.0.
    if example["is_noisy"]:
        return {"verdict": "wrong", "proposed_label": str(example["gold_label"]), "confidence": 1.0}
    return {"verdict": "correct", "proposed_label": None, "confidence": 0.5}


POLICIES = {
    "accept_all": accept_all,
    "always_ambiguous": always_ambiguous,
    "flag_all": flag_all,
    "random_verdict": random_verdict,
    "oracle": oracle,
}


def run(task, policy, episodes, episode_length, env):
    rows = []
    for seed in range(episodes):
        rng = random.Random(10_000 + seed)
        obs = env.reset(seed=seed)
        done = False
        while not done:
            example = env.current_examples[env.current_step]
            result = env.step(Action(example_id=obs.example_id, **policy(obs, example, rng)))
            obs, done = result.observation, result.done
        state = env.state()
        rows.append((state.cumulative_score, state.accept_all_score, state.oracle_score))
    return {
        "mean_cumulative": round(statistics.mean(r[0] for r in rows), 3),
        "stdev_cumulative": round(statistics.stdev(r[0] for r in rows), 3),
        "skill": None if (s := aggregate_skill(rows)) is None else round(s, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "reference_baselines.json"))
    args = parser.parse_args()

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    results = {
        "measured_at": datetime.date.today().isoformat(),
        "commit": commit or None,
        "episodes_per_cell": args.episodes,
        "episode_length": args.episode_length,
        "seeds": f"0..{args.episodes - 1}",
        "packages": {p: metadata.version(p) for p in ("datasets", "pydantic")},
        "tasks": {},
    }

    for task in TASKS:
        env = LabelingQAEnv(task=task, episode_length=args.episode_length)
        results["tasks"][task] = {}
        for name, policy in POLICIES.items():
            cell = run(task, policy, args.episodes, args.episode_length, env)
            results["tasks"][task][name] = cell
            print(f"{task:7s} {name:17s} cumulative {cell['mean_cumulative']:6.2f} "
                  f"(sd {cell['stdev_cumulative']:.2f})  skill {cell['skill']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
