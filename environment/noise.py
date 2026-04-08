import random

def inject_noise_task1(examples: list[dict]) -> list[dict]:
    """
    Task 1: gold_label is "0" or "1".
    Noise type: random flips - 20% of examples get their label flipped to the opposite.
    """
    random.seed(42)
    noised_examples = []
    
    for ex in examples:
        new_ex = ex.copy()
        gold_label = str(new_ex["gold_label"])
        
        # 20% chance to flip
        if random.random() < 0.20:
            ai_label = "1" if gold_label == "0" else "0"
            is_noisy = True
        else:
            ai_label = gold_label
            is_noisy = False
            
        new_ex["ai_label"] = ai_label
        new_ex["is_noisy"] = is_noisy
        noised_examples.append(new_ex)
        
    return noised_examples

def inject_noise_task2(examples: list[dict]) -> list[dict]:
    """
    Task 2: gold_label is "entailment", "neutral", or "contradiction".
    Noise type: systematic bias - AI always predicts "neutral" when the correct label 
    is "contradiction". Flips 15% of "entailment" to "neutral" too.
    """
    random.seed(42)
    noised_examples = []
    
    for ex in examples:
        new_ex = ex.copy()
        gold_label = str(new_ex["gold_label"])
        
        is_noisy = False
        ai_label = gold_label
        
        if gold_label == "contradiction":
            ai_label = "neutral"
            is_noisy = True
        elif gold_label == "entailment":
            if random.random() < 0.15:
                ai_label = "neutral"
                is_noisy = True
                
        new_ex["ai_label"] = ai_label
        new_ex["is_noisy"] = is_noisy
        noised_examples.append(new_ex)
        
    return noised_examples

def inject_noise_task3(examples: list[dict]) -> list[dict]:
    """
    Task 3: gold_label is int 0-13.
    Noise type: confident wrong labels on edge cases - for examples where gold_label 
    is in [0, 1, 2, 3], 30% chance of being mislabeled to a nearby category (+1 or -1). 
    For all others, 10% random flip to any other label.
    """
    random.seed(42)
    noised_examples = []
    
    for ex in examples:
        new_ex = ex.copy()
        gold_label = int(new_ex["gold_label"])
        
        is_noisy = False
        ai_label = gold_label
        
        if gold_label in [0, 1, 2, 3]:
            if random.random() < 0.30:
                is_noisy = True
                offset = random.choice([-1, 1])
                ai_label = max(0, min(13, gold_label + offset))
        else:
            if random.random() < 0.10:
                is_noisy = True
                possible_labels = [l for l in range(14) if l != gold_label]
                ai_label = random.choice(possible_labels)
                
        new_ex["ai_label"] = str(ai_label)
        new_ex["is_noisy"] = is_noisy
        noised_examples.append(new_ex)
        
    return noised_examples

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.loader import load_task1, load_task2, load_task3
    
    # Load
    print("Loading datasets...")
    data1 = load_task1()
    data2 = load_task2()
    data3 = load_task3()
    
    # Task 1
    print(f"\nTask 1 loaded: {len(data1)} examples")
    noised1 = inject_noise_task1(data1)
    num_noisy1 = sum(1 for ex in noised1 if ex["is_noisy"])
    print(f"Task 1 noised: {num_noisy1} / {len(noised1)}")
    noisy_example1 = next((ex for ex in noised1 if ex["is_noisy"]), None)
    if noisy_example1:
        print(f"Task 1 example: {noisy_example1}")
        
    # Task 2
    print(f"\nTask 2 loaded: {len(data2)} examples")
    noised2 = inject_noise_task2(data2)
    num_noisy2 = sum(1 for ex in noised2 if ex["is_noisy"])
    print(f"Task 2 noised: {num_noisy2} / {len(noised2)}")
    noisy_example2 = next((ex for ex in noised2 if ex["is_noisy"]), None)
    if noisy_example2:
        print(f"Task 2 example: {noisy_example2}")
        
    # Task 3
    print(f"\nTask 3 loaded: {len(data3)} examples")
    noised3 = inject_noise_task3(data3)
    num_noisy3 = sum(1 for ex in noised3 if ex["is_noisy"])
    print(f"Task 3 noised: {num_noisy3} / {len(noised3)}")
    noisy_example3 = next((ex for ex in noised3 if ex["is_noisy"]), None)
    if noisy_example3:
        print(f"Task 3 example: {noisy_example3}")
