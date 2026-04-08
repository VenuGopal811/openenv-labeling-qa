def grade_task1(action: dict, example: dict) -> dict:
    verdict = action.get("verdict")
    proposed_label = action.get("proposed_label")
    gold_label_str = str(example["gold_label"])
    is_noisy = example.get("is_noisy", False)
    
    score = 0.0
    reason = ""
    
    if verdict == "ambiguous":
        score = 0.3
        reason = "Partial credit for ambiguous (task 1 has no truly ambiguous cases)."
    elif is_noisy and verdict == "wrong":
        score = 1.0
        reason = "Correctly identified noisy label."
        if proposed_label == gold_label_str:
            score = min(1.0, score + 0.5)
            reason = "Correctly identified noisy label and proposed the correct label."
    elif is_noisy and verdict == "correct":
        score = 0.0
        reason = "Failed to identify noisy label."
    elif not is_noisy and verdict == "correct":
        score = 1.0
        reason = "Correctly accepted a valid label."
    elif not is_noisy and verdict == "wrong":
        score = -0.5
        reason = "Incorrectly rejected a valid label."
        
    return {"score": float(score), "reason": reason, "gold_label": gold_label_str}

def grade_task2(action: dict, example: dict) -> dict:
    verdict = action.get("verdict")
    proposed_label = action.get("proposed_label")
    gold_label_str = str(example["gold_label"])
    is_noisy = example.get("is_noisy", False)
    
    score = 0.0
    reason = ""
    
    if verdict == "ambiguous":
        if gold_label_str == "neutral":
            score = 0.5
            reason = "Correctly identified neutral/ambiguous case."
        else:
            score = -0.3
            reason = "Incorrectly labeled non-neutral case as ambiguous."
    elif is_noisy and verdict == "wrong":
        score = 1.0
        reason = "Correctly identified noisy label."
        if proposed_label == gold_label_str:
            score = min(1.0, score + 0.5)
            reason = "Correctly identified noisy label and proposed the correct label."
    elif is_noisy and verdict == "correct":
        score = 0.0
        reason = "Failed to identify noisy label."
    elif not is_noisy and verdict == "correct":
        score = 1.0
        reason = "Correctly accepted a valid label."
    elif not is_noisy and verdict == "wrong":
        score = -0.5
        reason = "Incorrectly rejected a valid label."
        
    return {"score": float(score), "reason": reason, "gold_label": gold_label_str}

def grade_task3(action: dict, example: dict) -> dict:
    verdict = action.get("verdict")
    proposed_label = action.get("proposed_label")
    confidence = action.get("confidence", 1.0)
    gold_label_str = str(example["gold_label"])
    is_noisy = example.get("is_noisy", False)
    
    score = 0.0
    reason = ""
    
    if verdict == "ambiguous":
        score = 0.4
        reason = "Partial credit for ambiguous (legal categories genuinely overlap)."
    elif is_noisy and verdict == "wrong":
        score = 1.0
        reason = "Correctly identified noisy label."
        if proposed_label == gold_label_str:
            score = min(1.0, score + 0.5)
            reason = "Correctly identified noisy label and proposed the correct label."
    elif is_noisy and verdict == "correct":
        if confidence > 0.8:
            score = -0.5
            reason = "Failed to identify noisy label and penalized for overconfidence."
        else:
            score = 0.0
            reason = "Failed to identify noisy label."
    elif not is_noisy and verdict == "correct":
        score = 1.0
        reason = "Correctly accepted a valid label."
    elif not is_noisy and verdict == "wrong":
        score = -0.5
        reason = "Incorrectly rejected a valid label."
        
    return {"score": float(score), "reason": reason, "gold_label": gold_label_str}

def grade(task: str, action: dict, example: dict) -> dict:
    """
    Unified entry point - routes to correct grader based on task ("easy", "medium", "hard").
    """
    if task == "easy":
        return grade_task1(action, example)
    elif task == "medium":
        return grade_task2(action, example)
    elif task == "hard":
        return grade_task3(action, example)
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    print("--- Task 1 (Easy) Sanity Check ---")
    ex1 = {"gold_label": 1, "ai_label": "0", "is_noisy": True}
    act1 = {"verdict": "wrong", "proposed_label": "1", "confidence": 0.9}
    res1 = grade("easy", act1, ex1)
    print(f"Action:  {act1}")
    print(f"Example: {ex1}")
    print(f"Result:  {res1}\n")
    
    print("--- Task 2 (Medium) Sanity Check ---")
    ex2 = {"gold_label": "neutral", "ai_label": "entailment", "is_noisy": True}
    act2 = {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5}
    res2 = grade("medium", act2, ex2)
    print(f"Action:  {act2}")
    print(f"Example: {ex2}")
    print(f"Result:  {res2}\n")
    
    print("--- Task 3 (Hard) Sanity Check ---")
    ex3 = {"gold_label": 11, "ai_label": "10", "is_noisy": True}
    act3 = {"verdict": "correct", "proposed_label": None, "confidence": 0.9}
    res3 = grade("hard", act3, ex3)
    print(f"Action:  {act3}")
    print(f"Example: {ex3}")
    print(f"Result:  {res3}\n")
