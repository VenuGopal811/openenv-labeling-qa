---
title: LabelSense OpenEnv
emoji: 🏷️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🏷️ LabelSense: AI-Assisted Data Labeling QA Environment

> An OpenEnv-compliant environment where an AI agent audits AI-generated labels on medical and legal text, identifies mislabeled examples, flags ambiguous cases, and proposes corrections.

---

## Why This Exists

AI models are increasingly used to auto-label training data at scale. The problem: label quality is inconsistent, and **confidently wrong labels corrupt downstream models silently**. No one catches them until the model is already in production.

LabelSense gives agents a structured environment to practice exactly this: reviewing batches of AI-generated labels, catching errors, and making calibrated decisions about ambiguity. It maps directly to a real MLOps workflow that every team doing data labeling at scale deals with daily.

---

## Environment Overview

The agent is shown examples from real medical and legal datasets, each pre-labeled by a simulated AI labeler (with injected noise). The agent must:

- Decide if each label is **correct**, **wrong**, or **ambiguous**
- If wrong, propose the **correct label**
- Signal **confidence** in its decision

The environment scores the agent on accuracy, calibration, and how responsibly it handles uncertainty.

---

## Datasets

| Task | Dataset | Source | Label Type |
|------|---------|--------|------------|
| Easy | Medical Question Pairs | `curaihealth/medical_questions_pairs` | Binary: similar / not similar |
| Medium | Stanford NLI | `snli` | 3-class: entailment / neutral / contradiction |
| Hard | SCOTUS (LexGLUE) | `coastalcph/lex_glue` (scotus) | 14-class: Supreme Court issue areas |

---

## Action & Observation Space

### Observation
What the agent sees at each step:

```json
{
  "example_id": "task1_042",
  "task": "easy",
  "input": {
    "text1": "Does ibuprofen reduce fever?",
    "text2": "Can ibuprofen be used to treat high temperature?"
  },
  "ai_label": "not_similar",
  "label_options": ["similar", "not_similar"]
}
```

### Action
What the agent responds with:

```json
{
  "example_id": "task1_042",
  "verdict": "wrong",
  "proposed_label": "similar",
  "confidence": 0.91
}
```

`verdict` must be one of: `correct`, `wrong`, `ambiguous`

---

## Reward Function

Per-step reward, taken from `environment/grader.py`:

| Agent action | Label is | Easy | Medium | Hard |
|---|---|---|---|---|
| `correct` (accept) | clean | +1.0 | +1.0 | +1.0 |
| `correct` (accept) | noisy | 0.0 | 0.0 | 0.0, or -0.5 if confidence > 0.8 |
| `wrong` (flag) | noisy | +0.7 | +0.7 | +0.7 |
| `wrong` plus the gold label as `proposed_label` | noisy | +1.0 | +1.0 | +1.0 |
| `wrong` (flag) | clean | -0.5 | -0.5 | -0.5 |
| `ambiguous` | any | +0.3 | +0.5 if gold is neutral, else -0.3 | +0.4 |

### Read the skill score, not the raw reward

Most labels are clean, and accepting a clean label pays +1.0. So a policy that approves every label without reading it earns a high raw score: it averages 8.035 out of 10 on the easy task (see the reference table below). Raw cumulative reward alone cannot tell a useful reviewer from one that does nothing.

`/grader` and `state()` therefore also report, for the exact examples in the episode:

- `accept_all_score`: what approving everything (confidence 0.5) would have scored
- `oracle_score`: the best achievable score
- `skill = (cumulative - accept_all) / (oracle - accept_all)`

`skill` is 0 for approve-all, 1 for a perfect reviewer, and negative when an agent does worse than approving everything. It is `null` for an episode that happens to contain no noisy labels. Across many episodes, pool the numerator and denominator rather than averaging per-episode ratios (`environment/scoring.py`, `aggregate_skill`).

---

## Tasks

### Task 1 (Easy): Medical Question Pair Similarity
**Dataset:** `curaihealth/medical_questions_pairs`
**Label type:** Binary (0 = not similar, 1 = similar)
**Noise:** ~20% random label flips on clear-cut examples
**What makes it easy:** Labels are mostly unambiguous. Errors are random, not systematic.

### Task 2 (Medium): Natural Language Inference
**Dataset:** `stanfordnlp/snli`
**Label type:** 3-class (entailment / neutral / contradiction)
**Noise:** Systematic bias. Every contradiction is labeled "neutral", and 15% of entailments are too
**What makes it medium:** Requires understanding sentence-level logic. Neutral vs. contradiction is a common confusion point.

### Task 3 (Hard): SCOTUS Legal Issue Classification
**Dataset:** `coastalcph/lex_glue` (scotus config)
**Label type:** 14-class Supreme Court issue areas
**Noise:** Confident wrong labels on edge cases, near-duplicate category confusion
**What makes it hard:** 14 overlapping legal categories. AI labeler is confidently wrong, not randomly wrong. Requires legal domain reasoning.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode, returns first observation. Optional `seed` makes the example draw reproducible |
| POST | `/step` | Submit an action, returns next observation + reward |
| GET | `/state` | Returns current episode state and progress |
| GET | `/tasks` | Lists all tasks and their action schemas |
| POST | `/grader` | Returns cumulative score, accept-all and oracle reference scores, and skill |
| POST | `/baseline` | Runs baseline inference script, returns scores for all 3 tasks |

---

## Setup & Usage

### Local (Python)

```bash
git clone https://github.com/VenuGopal811/openenv-labeling-qa
cd openenv-labeling-qa

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Start the API server
uvicorn server.app:app --port 7860
```

### Docker

```bash
docker build -t labelsense-env .
docker run -p 7860:7860 labelsense-env
```

### Tests

```bash
pytest tests
```

### Reference baselines (no API key)

```bash
python scripts/reference_baselines.py --episodes 200
```

Writes `results/reference_baselines.json`.

### LLM baseline

```bash
export GROQ_API_KEY=your_key_here
uvicorn server.app:app --port 7860   # separate terminal
python baseline.py --episodes 20
```

Writes `results/llm_baseline_<model>.json`. Set `LABELSENSE_API_URL` to run against the hosted Space instead.

---

## OpenEnv Spec Compliance

- Typed Pydantic models for `Observation`, `Action`, `Reward`
- `reset()` → returns clean initial observation
- `step(action)` → returns observation, reward, done, info
- `state()` → returns current episode state
- `openenv.yaml` with full metadata
- Validated with `openenv validate`

---

## Baseline Scores

### Reference policies

200 seeded episodes (seeds 0 to 199) of 10 steps per task, from `results/reference_baselines.json`. Cumulative is the mean raw reward out of 10.

| Policy | Easy cumulative | Easy skill | Medium cumulative | Medium skill | Hard cumulative | Hard skill |
|---|---|---|---|---|---|---|
| Accept every label | 8.035 | 0.000 | 5.935 | 0.000 | 7.725 | 0.000 |
| Always "ambiguous" | 3.000 | -2.562 | -0.456 | -1.572 | 4.000 | -1.637 |
| Flag every label | -2.053 | -5.134 | 0.496 | -1.338 | -2.219 | -4.371 |
| Random verdict | 2.861 | -2.633 | 1.851 | -1.005 | 3.065 | -2.048 |
| Oracle (reads gold labels) | 10.000 | 1.000 | 10.000 | 1.000 | 10.000 | 1.000 |

### LLM baseline

Not yet re-run under the current grader. Run `python baseline.py --episodes 20` and report `skill`.

An earlier single-episode run of `llama-3.1-8b-instant` scored a cumulative 4.30 (easy), 2.00 (medium) and 4.90 (hard). That run used the previous reward (flagging a noisy label paid 1.0 whether or not the fix was right), so it is not comparable with the table above. Even so, all three numbers are below what accepting every label scores, which is how the approve-all problem was found.

---

## Project Structure

```
labelsense-openenv/
├── environment/
│   ├── env.py          # Core environment: reset(), step(), state()
│   ├── models.py       # Pydantic models: Observation, Action, Reward
│   ├── grader.py       # Per-task scoring logic
│   ├── scoring.py      # Accept-all and oracle references, skill score
│   └── noise.py        # AI labeler noise injection
├── tasks/
│   ├── task1_easy.py
│   ├── task2_medium.py
│   └── task3_hard.py
├── data/
│   └── loader.py       # HuggingFace dataset loading + sampling
├── server/app.py       # FastAPI app + all endpoints
├── scripts/
│   └── reference_baselines.py  # Non-LLM reference policies
├── tests/
│   └── test_scoring.py # Grader and skill-score tests
├── results/            # Measured baseline outputs (JSON)
├── baseline.py         # LLM baseline (Groq), multi-seed
├── openenv.yaml        # OpenEnv spec metadata
├── Dockerfile
└── README.md
```

---

## License

MIT