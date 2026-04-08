import os
import sys
import json
import requests
from groq import Groq

API_URL = "http://localhost:8000"

def parse_json_response(text: str) -> dict:
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2:
                # Remove starting and ending markdown fences
                if lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
                    
        parsed = json.loads(text)
        
        return {
            "verdict": parsed.get("verdict", "ambiguous"),
            "proposed_label": parsed.get("proposed_label"),
            "confidence": float(parsed.get("confidence", 0.5))
        }
    except Exception:
        # Default back to ambiguous on parse failure
        return {"verdict": "ambiguous", "proposed_label": None, "confidence": 0.5}

def run_baseline() -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)
        
    client = Groq(api_key=api_key)
    model = "llama-3.1-8b-instant"
    
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    for task in tasks:
        session_id = f"baseline_{task}"
        
        # 1. Reset Env
        reset_res = requests.post(f"{API_URL}/reset", json={
            "task": task,
            "episode_length": 10,
            "session_id": session_id
        })
        reset_res.raise_for_status()
        obs = reset_res.json()
        
        done = False
        while not done:
            # 2. Extract inputs depending on task definition
            input_fields = obs.get("input", {})
            if task == "easy":
                input_text = f"Text 1: {input_fields.get('text1')}\nText 2: {input_fields.get('text2')}"
            elif task == "medium":
                input_text = f"Premise: {input_fields.get('premise')}\nHypothesis: {input_fields.get('hypothesis')}"
            else: # hard
                raw_text = input_fields.get("text", "")
                input_text = f"Text: {raw_text[:300]}"  # Truncated to 300 chars
                
            ai_label = obs.get("ai_label")
            label_options = obs.get("label_options")
            
            # 3. Create Model Prompt
            prompt = f"""You are an expert AI auditor verifying labels for a dataset.
Your task is to review the provided input and decide if the assigned 'AI Label' is correct, wrong, or ambiguous.

Input Examples:
{input_text}

AI Label: {ai_label}
Valid Label Options: {label_options}

Instructions:
Evaluate the AI Label against the Input Examples.
Respond in pure JSON format only with the following keys:
- "verdict": purely one of "correct", "wrong", or "ambiguous"
- "proposed_label": if the verdict is "wrong", provide the correct label from the Valid Label Options as a string. Otherwise, use null.
- "confidence": a float between 0.0 and 1.0 representing your confidence.

Example of valid response:
{{"verdict": "wrong", "proposed_label": "1", "confidence": 0.85}}
"""

            # 4. Invoke LLM
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You output JSON strictly."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=0.0
            )
            
            # 5. Parse output
            response_text = chat_completion.choices[0].message.content
            action_dict = parse_json_response(response_text)
            
            # 6. Step Env
            step_payload = {
                "session_id": session_id,
                "example_id": obs.get("example_id"),
                "verdict": action_dict["verdict"],
                "proposed_label": action_dict.get("proposed_label"),
                "confidence": action_dict.get("confidence")
            }
            
            step_res = requests.post(f"{API_URL}/step", json=step_payload)
            step_res.raise_for_status()
            step_data = step_res.json()
            
            done = step_data.get("done", True)
            if not done:
                obs = step_data.get("observation", {})
                
        # 7. Collect Final Grade
        grader_res = requests.post(f"{API_URL}/grader", params={"session_id": session_id})
        grader_res.raise_for_status()
        final_info = grader_res.json()
        
        results[task] = final_info.get("cumulative_score", 0.0)
        
    return results

if __name__ == "__main__":
    print("Running baseline evaluations... (This evaluates the Groq model against the live API)")
    scores = run_baseline()
    
    print("\n--- Baseline Results ---")
    total_score = 0.0
    for task, score in scores.items():
        print(f"Task: {task.capitalize():<10} | Cumulative Score: {score:>5.2f}")
        total_score += score
        
    avg_score = total_score / len(scores) if scores else 0.0
    print("-" * 35)
    print(f"Overall Average Score: {avg_score:>5.2f}")
