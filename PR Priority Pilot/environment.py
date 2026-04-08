import random
from typing import Tuple, Dict, List, Optional
from pydantic import BaseModel

class Observation(BaseModel):
    pr_title: str
    pr_description: str
    files_changed: int
    labels: List[str]
    author: str

class Action(BaseModel):
    priority: int

class State(BaseModel):
    observation: Optional[Observation]
    done: bool

# Three tasks, each with 3 PRs
TASKS = {
    "easy": [
        {"title": "Fix typo", "desc": "Typo fix", "files": 1, "labels": ["docs"], "author": "junior", "truth": 0},
        {"title": "Add feature", "desc": "New toggle", "files": 2, "labels": ["feature"], "author": "mid", "truth": 1},
        {"title": "Urgent fix", "desc": "Crash fix", "files": 1, "labels": ["urgent"], "author": "senior", "truth": 2}
    ],
    "medium": [
        {"title": "Security patch", "desc": "SQL injection", "files": 3, "labels": ["security"], "author": "sec", "truth": 2},
        {"title": "Refactor", "desc": "Code cleanup", "files": 12, "labels": ["refactor"], "author": "senior", "truth": 1},
        {"title": "UI update", "desc": "Button style", "files": 2, "labels": ["ui"], "author": "junior", "truth": 0}
    ],
    "hard": [
        {"title": "Hotfix payment", "desc": "Timeout", "files": 4, "labels": ["critical"], "author": "lead", "truth": 2},
        {"title": "DB migration", "desc": "Add columns", "files": 8, "labels": ["database"], "author": "backend", "truth": 1},
        {"title": "Dependency update", "desc": "Bump versions", "files": 15, "labels": ["deps"], "author": "bot", "truth": 0}
    ]
}

class CodeReviewEnv:
    def __init__(self):
        self.task = "easy"
        self.current = None
        self.done = False
        self.pool = TASKS["easy"]

    def set_task(self, difficulty: str):
        self.task = difficulty
        self.pool = TASKS[difficulty]

    def reset(self) -> Observation:
        self.current = random.choice(self.pool).copy()
        self.done = False
        return Observation(
            pr_title=self.current["title"],
            pr_description=self.current["desc"],
            files_changed=self.current["files"],
            labels=self.current["labels"],
            author=self.current["author"]
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Already done")
        pred = action.priority
        truth = self.current["truth"]
        # Base reward
        if pred == truth:
            base = 0.85
        elif abs(pred - truth) == 1:
            base = 0.55
        else:
            base = 0.25
        # Add a tiny, deterministic offset (never 0.0 or 1.0)
        offset = (hash(self.current["title"]) % 100) / 1000.0  # 0.000 to 0.099
        reward = base + offset
        # Clamp to safe range
        reward = max(0.01, min(0.99, reward))
        self.done = True
        # Return a new PR for the next step
        next_obs = self.reset()
        info = {"true_priority": truth, "explanation": "ok"}
        return next_obs, reward, self.done, info

    def state(self) -> State:
        if self.current:
            obs = Observation(
                pr_title=self.current["title"],
                pr_description=self.current["desc"],
                files_changed=self.current["files"],
                labels=self.current["labels"],
                author=self.current["author"]
            )
            return State(observation=obs, done=self.done)
        return State(observation=None, done=self.done)
