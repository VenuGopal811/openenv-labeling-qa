import os
import sys
import time
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

def build_prompt(task_name, obs):
    base_instruction = (
        "You are an expert data labeling quality assurance AI. "
        "Your job is to audit the provided 'ai_label' against the input and the 'label_options'. "
        "Respond ONLY with a valid JSON object. Do not include any other text, reasoning, or markdown formatting.\n"
        "Required JSON keys:\n"
        "- 'verdict': string, one of ['correct', 'wrong', 'ambiguous'].\n"
        "- 'proposed_label': string, the correct label if verdict is 'wrong', or null if correct/ambiguous.\n"
        "- 'confidence': float, between 0.0 and 1.0.\n\n"
    )

    input_data = obs.get("input", {})
    ai_label = obs.get("ai_label")
    label_options = obs.get("label_options", [])

    if task_name == "easy":
        text1 = input_data.get("sentence1", input_data.get("text1", ""))
        text2 = input_data.get("sentence2", input_data.get("text2", ""))
        prompt = f"Task: Easy - Binary similarity.\nText 1: {text1}\nText 2: {text2}\nAI Label: {ai_label}\nLabel Options: {label_options}"
    elif task_name == "medium":
        premise = input_data.get("premise", "")
        hypothesis = input_data.get("hypothesis", "")
        prompt = f"Task: Medium - NLI.\nPremise: {premise}\nHypothesis: {hypothesis}\nAI Label: {ai_label}\nLabel Options: {label_options}"
    elif task_name == "hard":
        text = input_data.get("text", "")[:300]
        prompt = f"Task: Hard - SCOTUS legal issue classification.\nText (truncated to 300 chars): {text}...\nAI Label: {ai_label}\nLabel Options: {label_options}"
    else:
        prompt = f"Task: {task_name}\nInput: {input_data}\nAI Label: {ai_label}\nLabel Options: {label_options}"

    return base_instruction + prompt

def run_baseline() -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["easy", "medium", "hard"]
    
    global_start_time = time.time()
    
    scores = {}
    
    for task in tasks:
        session_id = f"inference_{task}"
        
        try:
            start_res = requests.post(
                "http://localhost:7860/reset", 
                json={"task": task, "episode_length": 10, "session_id": session_id}
            )
            start_res.raise_for_status()
            env_state = start_res.json()
        except Exception as e:
            print(f"Error resetting environment for task {task}: {e}")
            scores[task] = 0.0
            continue
            
        obs = env_state.get("observation", {})
        done = env_state.get("done", False)
        
        while not done:
            if time.time() - global_start_time > 18 * 60:
                print("Time limit exceeding 18 minutes. Breaking early.")
                break
                
            prompt = build_prompt(task, obs)
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                
                raw_content = response.choices[0].message.content.strip()
                
                # Strip markdown fences if present
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:]
                elif raw_content.startswith("```"):
                    raw_content = raw_content[3:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                    
                raw_content = raw_content.strip()
                parsed_action = json.loads(raw_content)
                
                action = {
                    "example_id": obs.get("example_id"),
                    "verdict": parsed_action.get("verdict", "ambiguous"),
                    "proposed_label": str(parsed_action.get("proposed_label")) if parsed_action.get("proposed_label") is not None else None,
                    "confidence": float(parsed_action.get("confidence", 0.5))
                }
            except Exception as e:
                action = {
                    "example_id": obs.get("example_id"),
                    "verdict": "ambiguous",
                    "proposed_label": None,
                    "confidence": 0.5
                }
            
            payload = {
                "session_id": session_id,
                "action": action
            }
            
            try:
                step_res = requests.post("http://localhost:7860/step", json=payload)
                step_res.raise_for_status()
                env_state = step_res.json()
                
                obs = env_state.get("observation", {})
                done = env_state.get("done", False)
            except Exception as e:
                print(f"Error stepping environment: {e}")
                break
                
        # Get final score
        try:
            grader_res = requests.post("http://localhost:7860/grader", json={"session_id": session_id})
            grader_res.raise_for_status()
            score = grader_res.json().get("score", 0.0)
            scores[task] = score
        except Exception as e:
            print(f"Error getting score for task {task}: {e}")
            scores[task] = 0.0
            
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    scores["average"] = avg_score
    return scores

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it to run the baseline evaluation.", file=sys.stderr)
        sys.exit(1)
        
    print("Running LabelSense Baseline...")
    results = run_baseline()
    
    print("\n=== LabelSense Baseline Results ===")
    print(f"Task: Easy    | Score: {results.get('easy', 0.0):.2f}")
    print(f"Task: Medium  | Score: {results.get('medium', 0.0):.2f}")
    print(f"Task: Hard    | Score: {results.get('hard', 0.0):.2f}")
    print(f"Average Score: {results.get('average', 0.0):.2f}")
