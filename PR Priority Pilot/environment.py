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

# Three tasks (easy, medium, hard) each with 3 examples
TASKS = {
    "easy": [
        {"title": "Fix typo in README", "desc": "Corrected spelling", "files": 1, "labels": ["docs"], "author": "junior", "truth": 0},
        {"title": "Add new feature toggle", "desc": "Minor feature", "files": 2, "labels": ["feature"], "author": "mid", "truth": 1},
        {"title": "URGENT: Fix login crash", "desc": "Hotfix for production", "files": 1, "labels": ["bug","urgent"], "author": "senior", "truth": 2}
    ],
    "medium": [
        {"title": "Security patch for SQL injection", "desc": "Critical vulnerability", "files": 3, "labels": ["security"], "author": "sec", "truth": 2},
        {"title": "Refactor logging module", "desc": "Code cleanup", "files": 12, "labels": ["refactor"], "author": "senior", "truth": 1},
        {"title": "Update button styles", "desc": "UI tweak", "files": 2, "labels": ["ui"], "author": "junior", "truth": 0}
    ],
    "hard": [
        {"title": "HOTFIX: Payment gateway timeout", "desc": "Customers cannot pay", "files": 4, "labels": ["critical","bug"], "author": "lead", "truth": 2},
        {"title": "Migrate database schema", "desc": "Add columns, no downtime", "files": 8, "labels": ["database"], "author": "backend", "truth": 1},
        {"title": "Update dependencies", "desc": "Regular maintenance", "files": 15, "labels": ["dependencies"], "author": "bot", "truth": 0}
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
            raise RuntimeError("Episode already done")
        pred = action.priority
        truth = self.current["truth"]
        # Base reward strictly between 0 and 1
        if pred == truth:
            base = 0.8
        elif abs(pred - truth) == 1:
            base = 0.5
        else:
            base = 0.2
        # Add tiny deterministic noise to avoid exact 0.0/1.0
        noise = (hash(self.current["title"]) % 10) / 100.0  # 0.00 to 0.09
        reward = base + noise
        # Clamp to (0.01, 0.99)
        reward = max(0.01, min(0.99, reward))
        self.done = True
        # Return a new observation (next PR)
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
