from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from typing import Dict, Optional

from environment.env import LabelingQAEnv
from environment.models import Action

app = FastAPI(title="LabelSense OpenEnv")

envs: Dict[str, LabelingQAEnv] = {}

class ResetRequest(BaseModel):
    task: str = "easy"
    episode_length: int = 10
    session_id: str = "default"

class StepRequest(BaseModel):
    session_id: str = "default"
    example_id: str
    verdict: str
    proposed_label: str | None = None
    confidence: float = 0.8

@app.post("/reset")
async def reset_endpoint(request: Request):
    # 1. Safely try to read the raw JSON body
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    # 2. If the grader literally sent 'null', force it to an empty dictionary
    if body is None:
        body = {}
        
    # 3. Manually pull out our variables with safe defaults
    task = body.get("task", "easy")
    episode_length = body.get("episode_length", 10)
    session_id = body.get("session_id", "default")
        
    try:
        env = LabelingQAEnv(task=task, episode_length=episode_length)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    envs[session_id] = env
    
    try:
        obs = env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
@app.post("/step")
def step_endpoint(req: StepRequest):
    if req.session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = envs[req.session_id]
    
    try:
        action = Action(
            example_id=req.example_id,
            verdict=req.verdict,
            proposed_label=req.proposed_label,
            confidence=req.confidence
        )
        res = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return res.model_dump() if hasattr(res, 'model_dump') else res.dict()

@app.get("/state")
def state_endpoint(session_id: str = "default"):
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = envs[session_id]
    state = env.state()
    return state.model_dump() if hasattr(state, 'model_dump') else state.dict()

@app.get("/tasks")
def tasks_endpoint():
    schema = {
        "session_id": "str",
        "example_id": "str",
        "verdict": "str",
        "proposed_label": "str | None",
        "confidence": "float"
    }
    
    return [
        {
            "name": "easy",
            "difficulty": "Easy",
            "description": "Binary classification setup on medical subsets",
            "action_schema": schema
        },
        {
            "name": "medium",
            "difficulty": "Medium",
            "description": "NLI textual entailment configuration",
            "action_schema": schema
        },
        {
            "name": "hard",
            "difficulty": "Hard",
            "description": "Complex multi-label edge case tagging",
            "action_schema": schema
        }
    ]

@app.post("/grader")
def grader_endpoint(session_id: str = "default"):
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = envs[session_id]
    state = env.state()
    
    return {
        "session_id": session_id,
        "task": state.task,
        "cumulative_score": state.cumulative_score,
        "total_steps": state.total_steps,
        "done": state.done
    }

@app.post("/baseline")
def baseline_endpoint():
    try:
        from baseline import run_baseline
        return run_baseline()
    except ImportError:
        return {"status": "baseline not yet implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_endpoint():
    return {"status": "ok", "service": "LabelSense OpenEnv"}

import uvicorn

def main():
    """Entry point for the OpenEnv validator."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()