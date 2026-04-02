# 🏷️ LabelSense — AI-Assisted Data Labeling QA Environment

> An OpenEnv-compliant environment where an AI agent audits AI-generated labels on medical and legal text, identifies mislabeled examples, flags ambiguous cases, and proposes corrections.

---

## Why This Exists

AI models are increasingly used to auto-label training data at scale. The problem: label quality is inconsistent, and **confidently wrong labels corrupt downstream models silently**. No one catches them until the model is already in production.

LabelSense gives agents a structured environment to practice exactly this — reviewing batches of AI-generated labels, catching errors, and making calibrated decisions about ambiguity. It maps directly to a real MLOps workflow that every team doing data labeling at scale deals with daily.

---

## Environment Overview

The agent is shown examples from real medical and legal datasets, each pre-labeled by a simulated AI labeler (with injected noise). The agent must:

- Decide if each label is **correct**, **wrong**, or **ambiguous**
- If wrong — propose the **correct label**
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

| Agent Action | Condition | Reward |
|---|---|---|
| Flags label as wrong | Label is actually wrong | +1.0 |
| Proposes correct fix | Fix matches gold label | +0.5 bonus |
| Flags as ambiguous | Example is genuinely ambiguous | +0.7 |
| Flags as ambiguous | Example is actually clear | -0.3 |
| Marks wrong label as correct | Misses the error | 0.0 |
| Proposes wrong fix confidently | High confidence, wrong answer | -0.5 |

Rewards are designed to encourage **calibrated uncertainty** — an agent that admits it doesn't know scores better than one that guesses confidently and gets it wrong.

---

## Tasks

### Task 1 — Easy: Medical Question Pair Similarity
**Dataset:** `curaihealth/medical_questions_pairs`
**Label type:** Binary (0 = not similar, 1 = similar)
**Noise:** ~20% random label flips on clear-cut examples
**Expected agent score:** 0.75 – 0.90
**What makes it easy:** Labels are mostly unambiguous. Errors are random, not systematic.

### Task 2 — Medium: Natural Language Inference
**Dataset:** `snli`
**Label type:** 3-class (entailment / neutral / contradiction)
**Noise:** Systematic bias — AI labeler over-predicts "neutral" when uncertain
**Expected agent score:** 0.55 – 0.75
**What makes it medium:** Requires understanding sentence-level logic. Neutral vs. contradiction is a common confusion point.

### Task 3 — Hard: SCOTUS Legal Issue Classification
**Dataset:** `coastalcph/lex_glue` (scotus config)
**Label type:** 14-class Supreme Court issue areas
**Noise:** Confident wrong labels on edge cases, near-duplicate category confusion
**Expected agent score:** 0.30 – 0.55
**What makes it hard:** 14 overlapping legal categories. AI labeler is confidently wrong, not randomly wrong. Requires legal domain reasoning.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode, returns first observation |
| POST | `/step` | Submit an action, returns next observation + reward |
| GET | `/state` | Returns current episode state and progress |
| GET | `/tasks` | Lists all tasks and their action schemas |
| POST | `/grader` | Returns grader score after episode completes |
| POST | `/baseline` | Runs baseline inference script, returns scores for all 3 tasks |

---

## Setup & Usage

### Local (Python)

```bash
# Clone the repo
git clone https://github.com/yourusername/labelsense-openenv
cd labelsense-openenv

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload
```

### Docker

```bash
docker build -t labelsense-env .
docker run -p 8000:8000 labelsense-env
```

### Run Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline.py
```

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

| Task | Model | Cumulative Score (10 steps) |
|------|-------|-----------------------------|
| Easy (medical pairs) | llama-3.1-8b-instant | 4.30 |
| Medium (NLI) | llama-3.1-8b-instant | 2.00 |
| Hard (SCOTUS) | llama-3.1-8b-instant | 4.90 |
| **Overall Average** | | **3.73** |

*Scores represent cumulative reward over 10 steps per task. Maximum possible score per task is 10.0 (all correct with fixes). Scores above 0 indicate the agent performs better than random.*

---

## Project Structure

```
labelsense-openenv/
├── environment/
│   ├── env.py          # Core environment — reset(), step(), state()
│   ├── models.py       # Pydantic models: Observation, Action, Reward
│   ├── grader.py       # Per-task scoring logic
│   └── noise.py        # AI labeler noise injection
├── tasks/
│   ├── task1_easy.py
│   ├── task2_medium.py
│   └── task3_hard.py
├── data/
│   └── loader.py       # HuggingFace dataset loading + sampling
├── main.py             # FastAPI app + all endpoints
├── baseline.py         # Baseline inference script (OpenAI API)
├── openenv.yaml        # OpenEnv spec metadata
├── Dockerfile
└── README.md
```

---

## License

MIT