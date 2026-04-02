import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.models import Observation, Action, Reward, StepResult, EpisodeState
from environment.grader import grade
from data.loader import load_task1, load_task2, load_task3
from environment.noise import inject_noise_task1, inject_noise_task2, inject_noise_task3

class LabelingQAEnv:
    def __init__(self, task: str = "easy", episode_length: int = 10):
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"Task must be one of 'easy', 'medium', 'hard'. Got {task}")
            
        self.task = task
        self.episode_length = episode_length
        
        if task == "easy":
            data = load_task1()
            self.examples = inject_noise_task1(data)
        elif task == "medium":
            data = load_task2()
            self.examples = inject_noise_task2(data)
        elif task == "hard":
            data = load_task3()
            self.examples = inject_noise_task3(data)
            
        self.current_step = 0
        self.cumulative_score = 0.0
        self.done = False
        self.current_examples = []
        
    def _build_observation(self, example: dict) -> Observation:
        if self.task == "easy":
            input_dict = {
                "text1": example.get("text1", ""),
                "text2": example.get("text2", "")
            }
            label_options = ["0", "1"]
        elif self.task == "medium":
            input_dict = {
                "premise": example.get("premise", ""),
                "hypothesis": example.get("hypothesis", "")
            }
            label_options = ["entailment", "neutral", "contradiction"]
        else: # hard
            text = example.get("text", "")
            input_dict = {"text": text[:500]}
            label_options = [str(i) for i in range(14)]
            
        return Observation(
            example_id=str(example["id"]),
            task=self.task,
            input=input_dict,
            ai_label=str(example["ai_label"]),
            label_options=label_options,
            episode_step=self.current_step,
            total_steps=self.episode_length
        )

    def reset(self) -> Observation:
        if self.episode_length > len(self.examples):
            raise ValueError("Episode length exceeds available examples.")
            
        self.current_examples = random.sample(self.examples, self.episode_length)
        self.current_step = 0
        self.cumulative_score = 0.0
        self.done = False
        
        return self._build_observation(self.current_examples[0])

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is already done.")
            
        current_example = self.current_examples[self.current_step]
        
        # Pydantic v2 compatible dict dump
        action_dict = action.model_dump() if hasattr(action, 'model_dump') else action.dict()
        score_dict = grade(self.task, action_dict, current_example)
        
        reward = Reward(
            example_id=str(current_example["id"]),
            score=score_dict["score"],
            reason=score_dict["reason"],
            gold_label=str(score_dict["gold_label"])
        )
        
        self.current_step += 1
        self.cumulative_score += reward.score
        
        if self.current_step >= self.episode_length:
            self.done = True
            
        obs = None if self.done else self._build_observation(self.current_examples[self.current_step])
        info = {
            "cumulative_score": self.cumulative_score,
            "step": self.current_step
        }
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info=info
        )

    def state(self) -> EpisodeState:
        return EpisodeState(
            task=self.task,
            current_step=self.current_step,
            total_steps=self.episode_length,
            cumulative_score=self.cumulative_score,
            done=self.done
        )


if __name__ == "__main__":
    import sys
    import os
    # Dynamically inject root path specifically inside __main__ execution for simplicity
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tasks = ["easy", "medium", "hard"]
    for t in tasks:
        print(f"\n{'='*50}")
        print(f"Testing LabelingQAEnv(task='{t}')")
        print(f"{'='*50}")
        
        # Keep episode length short for testing
        env = LabelingQAEnv(task=t, episode_length=3)
        obs = env.reset()
        
        for i in range(3):
            print(f"\n--- Step {i+1} ---")
            
            # Safely check for model_dump or standard dict wrapper
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
            print(f"Observation: {obs_dict}")
            
            first_option = obs.label_options[0]
            action = Action(
                example_id=obs.example_id,
                verdict="wrong",
                proposed_label=first_option,
                confidence=0.8
            )
            
            res = env.step(action)
            reward_dict = res.reward.model_dump() if hasattr(res.reward, 'model_dump') else res.reward.dict()
            
            print(f"Reward: {reward_dict}")
            print(f"Cumulative Score: {env.cumulative_score}")
            
            obs = res.observation
