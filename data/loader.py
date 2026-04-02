"""
data/loader.py - HuggingFace dataset loading for OpenEnv Labeling QA.

Loads three classification datasets, samples 150 examples from each,
and returns them as clean Python lists of dicts with standardized keys.

Dependencies: pip install datasets
"""

import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 150
SEED = 42

# NLI label mapping (int -> str) used by bigbio NLI datasets
_NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hf_dataset(path: str, split: str = "train", name: str = None):
    """
    Wrapper around datasets.load_dataset that handles trust_remote_code
    gracefully across different versions of the `datasets` library.
    """
    from datasets import load_dataset
    import inspect

    kwargs = {"path": path, "split": split}
    if name is not None:
        kwargs["name"] = name

    # Only pass trust_remote_code if the installed version supports it
    sig = inspect.signature(load_dataset)
    if "trust_remote_code" in sig.parameters:
        kwargs["trust_remote_code"] = True

    return load_dataset(**kwargs)


def _sample(dataset, n: int, seed: int = SEED):
    """Return a random sample of *n* rows from a HuggingFace Dataset."""
    if len(dataset) <= n:
        return dataset
    return dataset.shuffle(seed=seed).select(range(n))


def _safe_str(value) -> str:
    """Convert a value to a stripped string, handling None gracefully."""
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Task 1 - Medical Question Pairs (binary: 0 / 1)
# Dataset: curaihealth/medical_questions_pairs
# ---------------------------------------------------------------------------

def load_task1() -> list[dict]:
    """
    Load the curaihealth/medical_questions_pairs dataset.

    Returns a list of 150 dicts:
        {id: str, text1: str, text2: str, gold_label: int}
    where gold_label is 0 or 1.
    """
    try:
        print("[Task 1] Loading curaihealth/medical_questions_pairs ...")
        ds = _load_hf_dataset("curaihealth/medical_questions_pairs", split="train")

        sampled = _sample(ds, SAMPLE_SIZE)

        results: list[dict] = []
        for idx, row in enumerate(sampled):
            results.append({
                "id": f"task1_{idx}",
                "text1": _safe_str(row.get("question_1", row.get("question1", ""))),
                "text2": _safe_str(row.get("question_2", row.get("question2", ""))),
                "gold_label": int(row.get("label", 0)),
            })

        print(f"[Task 1] OK - Loaded {len(results)} examples.")
        return results

    except Exception as exc:
        print(f"[Task 1] FAILED - {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Task 2 - NLI (3-class: entailment / neutral / contradiction)
# Dataset: snli
# ---------------------------------------------------------------------------

def load_task2() -> list[dict]:
    """
    Load the Stanford NLI (snli) dataset.

    Filters out unlabeled examples (label == -1), then samples 150.

    Returns a list of 150 dicts:
        {id: str, premise: str, hypothesis: str, gold_label: str}
    where gold_label is "entailment", "neutral", or "contradiction".
    """
    try:
        print("[Task 2] Loading snli ...")
        ds = _load_hf_dataset("snli", split="train")

        # SNLI contains some unlabeled rows marked with label == -1
        ds = ds.filter(lambda x: x["label"] != -1)

        sampled = _sample(ds, SAMPLE_SIZE)

        results: list[dict] = []
        for idx, row in enumerate(sampled):
            label_str = _NLI_LABEL_MAP.get(row["label"], str(row["label"]))
            results.append({
                "id": f"task2_{idx}",
                "premise": _safe_str(row.get("premise", "")),
                "hypothesis": _safe_str(row.get("hypothesis", "")),
                "gold_label": label_str,
            })

        print(f"[Task 2] OK - Loaded {len(results)} examples.")
        return results

    except Exception as exc:
        print(f"[Task 2] FAILED - {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Task 3 - SCOTUS Legal Classification (14-class: 0-13)
# Dataset: coastalcph/lex_glue  config="scotus"
# ---------------------------------------------------------------------------

def load_task3() -> list[dict]:
    """
    Load the coastalcph/lex_glue (scotus) dataset.

    Returns a list of 150 dicts:
        {id: str, text: str, gold_label: int}
    where gold_label is an int 0-13.
    """
    try:
        print("[Task 3] Loading coastalcph/lex_glue (scotus) ...")
        ds = _load_hf_dataset("coastalcph/lex_glue", split="train",
                              name="scotus")

        sampled = _sample(ds, SAMPLE_SIZE)

        results: list[dict] = []
        for idx, row in enumerate(sampled):
            results.append({
                "id": f"task3_{idx}",
                "text": _safe_str(row.get("text", "")),
                "gold_label": int(row.get("label", 0)),
            })

        print(f"[Task 3] OK - Loaded {len(results)} examples.")
        return results

    except Exception as exc:
        print(f"[Task 3] FAILED - {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Main - quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  OpenEnv Labeling QA - Dataset Loader Smoke Test")
    print("=" * 60)

    # -- Task 1 ---------------------------------------------------------------
    try:
        t1 = load_task1()
        print(f"\n[Sample] Task 1 ({len(t1)} total):")
        print(f"   {t1[0]}\n")
    except Exception as e:
        print(f"\n[ERROR] Task 1: {e}\n")

    # -- Task 2 ---------------------------------------------------------------
    try:
        t2 = load_task2()
        print(f"[Sample] Task 2 ({len(t2)} total):")
        print(f"   {t2[0]}\n")
    except Exception as e:
        print(f"\n[ERROR] Task 2: {e}\n")

    # -- Task 3 ---------------------------------------------------------------
    try:
        t3 = load_task3()
        print(f"[Sample] Task 3 ({len(t3)} total):")
        print(f"   {t3[0]}\n")
    except Exception as e:
        print(f"\n[ERROR] Task 3: {e}\n")

    print("=" * 60)
    print("  Done.")
    print("=" * 60)
