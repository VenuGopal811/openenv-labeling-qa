from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Observation(BaseModel):
    """What the agent observes at each step in the environment."""
    example_id: str
    task: Literal["easy", "medium", "hard"]
    input: dict
    ai_label: str
    label_options: list[str]
    episode_step: int
    total_steps: int

class Action(BaseModel):
    """The action the agent takes for a given observation."""
    example_id: str
    verdict: Literal["correct", "wrong", "ambiguous"]
    proposed_label: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_proposed_label(self) -> 'Action':
        if self.verdict == "wrong" and self.proposed_label is None:
            raise ValueError('proposed_label must be provided when verdict is "wrong"')
        return self

class Reward(BaseModel):
    """The reward given after an action is taken."""
    example_id: str
    score: float = Field(ge=-1.0, le=1.0)
    reason: str
    gold_label: str

class StepResult(BaseModel):
    """The full return value of taking a step in the environment."""
    observation: Observation | None
    reward: Reward
    done: bool
    info: dict

class EpisodeState(BaseModel):
    """The overall state of the current episode."""
    task: str
    current_step: int
    total_steps: int
    cumulative_score: float
    done: bool
