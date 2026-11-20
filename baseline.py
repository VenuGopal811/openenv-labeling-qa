"""
LLM baseline: runs a Groq model through the environment over several seeded
episodes per task and reports both raw cumulative reward and the skill score
(see environment/scoring.py). Read skill, not raw reward: skill 0 means the
model is no better than approving every label.

    export GROQ_API_KEY=...
    uvicorn server.app:app --port 7860      # in another terminal
    python baseline.py --episodes 20

Writes results/llm_baseline_<model>.json.
"""

import argparse
import datetime
import json
import os
import statistics
import sys

import requests
from groq import Groq

API_URL = os.environ.get("LABELSENSE_API_URL", "http://localhost:7860")
DEFAULT_MODEL = "llama-3.1-8b-instant"
TASKS = ["easy", "medium", "hard"]


def parse_json_response(text: str) -> dict:
    """Parse the model's JSON. Anything unusable becomes an 'ambiguous' verdict."""
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2:
                if lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])

        parsed = json.loads(text)
        verdict = parsed.get("verdict", "ambiguous")
        proposed = parsed.get("proposed_label")
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        if verdict not in ("correct", "wrong", "ambiguous") or (verdict == "wrong" and proposed is None):
            return {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5, "invalid": True}
        return {"verdict": verdict, "proposed_label": None if proposed is None else str(proposed),
                "confidence": confidence, "invalid": False}
    except Exception:
        return {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5, "invalid": True}


def build_prompt(task: str, obs: dict) -> str:
    fields = obs.get("input", {})
    if task == "easy":
        input_text = f"Text 1: {fields.get('text1')}\nText 2: {fields.get('text2')}"
    elif task == "medium":
        input_text = f"Premise: {fields.get('premise')}\nHypothesis: {fields.get('hypothesis')}"
    else:
        input_text = f"Text: {fields.get('text', '')[:300]}"

    return f"""You are an expert AI auditor verifying labels for a dataset.
Your task is to review the provided input and decide if the assigned 'AI Label' is correct, wrong, or ambiguous.

Input Examples:
{input_text}

AI Label: {obs.get("ai_label")}
Valid Label Options: {obs.get("label_options")}

Instructions:
Evaluate the AI Label against the Input Examples.
Respond in pure JSON format only with the following keys:
- "verdict": purely one of "correct", "wrong", or "ambiguous"
- "proposed_label": if the verdict is "wrong", provide the correct label from the Valid Label Options as a string. Otherwise, use null.
- "confidence": a float between 0.0 and 1.0 representing your confidence.

Example of valid response:
{{"verdict": "wrong", "proposed_label": "1", "confidence": 0.85}}
"""


def run_episode(client, model: str, task: str, seed: int, episode_length: int) -> dict:
    session_id = f"baseline_{task}_{seed}"
    res = requests.post(f"{API_URL}/reset", json={
        "task": task, "episode_length": episode_length, "session_id": session_id, "seed": seed,
    })
    res.raise_for_status()
    obs = res.json()

    invalid = 0
    done = False
    while not done:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output JSON strictly."},
                {"role": "user", "content": build_prompt(task, obs)},
            ],
            model=model,
            temperature=0.0,
        )
        action = parse_json_response(completion.choices[0].message.content)
        invalid += action.pop("invalid")

        step = requests.post(f"{API_URL}/step", json={
            "session_id": session_id, "example_id": obs.get("example_id"), **action,
        })
        step.raise_for_status()
        data = step.json()
        done = data.get("done", True)
        if not done:
            obs = data.get("observation", {})

    grade = requests.post(f"{API_URL}/grader", params={"session_id": session_id})
    grade.raise_for_status()
    g = grade.json()
    return {
        "seed": seed,
        "cumulative_score": g["cumulative_score"],
        "accept_all_score": g["accept_all_score"],
        "oracle_score": g["oracle_score"],
        "skill": g["skill"],
        "invalid_responses": invalid,
    }


def summarize(episodes: list[dict]) -> dict:
    gain = sum(e["cumulative_score"] - e["accept_all_score"] for e in episodes)
    headroom = sum(e["oracle_score"] - e["accept_all_score"] for e in episodes)
    scores = [e["cumulative_score"] for e in episodes]
    return {
        "episodes": len(episodes),
        "mean_cumulative": round(statistics.mean(scores), 3),
        "stdev_cumulative": round(statistics.stdev(scores), 3) if len(scores) > 1 else None,
        "mean_accept_all": round(statistics.mean(e["accept_all_score"] for e in episodes), 3),
        "skill": round(gain / headroom, 3) if headroom > 0 else None,
        "invalid_responses": sum(e["invalid_responses"] for e in episodes),
    }


def run_baseline(model: str = DEFAULT_MODEL, episodes: int = 3, episode_length: int = 10) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)
    client = Groq(api_key=api_key)

    results = {}
    for task in TASKS:
        rows = [run_episode(client, model, task, seed, episode_length) for seed in range(episodes)]
        results[task] = {"summary": summarize(rows), "episodes": rows}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=10)
    args = parser.parse_args()

    print(f"Running {args.model}: {args.episodes} episodes x {args.episode_length} steps per task against {API_URL}")
    results = run_baseline(args.model, args.episodes, args.episode_length)

    print(f"\n{'Task':<8}{'Cumulative':>12}{'Accept-all':>12}{'Skill':>8}{'Invalid':>9}")
    for task, r in results.items():
        s = r["summary"]
        print(f"{task:<8}{s['mean_cumulative']:>12.2f}{s['mean_accept_all']:>12.2f}{str(s['skill']):>8}{s['invalid_responses']:>9}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       f"llm_baseline_{args.model.replace('/', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"measured_at": datetime.date.today().isoformat(), "model": args.model,
                   "episode_length": args.episode_length, "seeds": f"0..{args.episodes - 1}",
                   "tasks": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
