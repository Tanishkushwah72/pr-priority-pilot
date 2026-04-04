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
    priority: int   # 0=Low, 1=Medium, 2=High

class State(BaseModel):
    observation: Optional[Observation]
    done: bool

EASY_PRS = [
    {"title": "Fix typo in README", "description": "Just corrected a spelling mistake.", "files_changed": 1, "labels": ["docs"], "author": "junior_dev", "true_priority": 0},
    {"title": "Update API endpoint for user profile", "description": "Minor change, no breaking changes.", "files_changed": 2, "labels": ["feature"], "author": "mid_dev", "true_priority": 1},
    {"title": "URGENT: Fix login crash on production", "description": "Hotfix for login error affecting all users.", "files_changed": 1, "labels": ["bug", "urgent"], "author": "senior_dev", "true_priority": 2}
]

MEDIUM_PRS = [
    {"title": "Security patch for SQL injection", "description": "Vulnerability in user input sanitization.", "files_changed": 3, "labels": ["security"], "author": "security_team", "true_priority": 2},
    {"title": "Refactor logging module", "description": "No functional change, just better structure.", "files_changed": 12, "labels": ["refactor"], "author": "senior_dev", "true_priority": 1},
    {"title": "Add new button to homepage", "description": "UI improvement, not urgent.", "files_changed": 2, "labels": ["ui", "feature"], "author": "junior_dev", "true_priority": 0}
]

HARD_PRS = [
    {"title": "HOTFIX: Payment gateway timeout", "description": "Customers cannot checkout. Needs immediate deploy.", "files_changed": 4, "labels": ["critical", "bug"], "author": "lead_dev", "true_priority": 2},
    {"title": "Migrate database schema (non-breaking)", "description": "Add new columns, no downtime.", "files_changed": 8, "labels": ["database"], "author": "backend_team", "true_priority": 1},
    {"title": "Update dependency versions", "description": "Regular maintenance, no urgency.", "files_changed": 15, "labels": ["dependencies"], "author": "bot", "true_priority": 0}
]

TASK_POOLS = {"easy": EASY_PRS, "medium": MEDIUM_PRS, "hard": HARD_PRS}

class CodeReviewEnv:
    def __init__(self):
        self.task_name = "easy"
        self.current_pr = None
        self.done = False
        self.task_pool = TASK_POOLS["easy"]

    def set_task(self, difficulty: str):
        self.task_name = difficulty
        self.task_pool = TASK_POOLS[difficulty]

    def reset(self) -> Observation:
        self.current_pr = random.choice(self.task_pool).copy()
        self.done = False
        return Observation(
            pr_title=self.current_pr["title"],
            pr_description=self.current_pr["description"],
            files_changed=self.current_pr["files_changed"],
            labels=self.current_pr["labels"],
            author=self.current_pr["author"]
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode done")
        pred = action.priority
        true_priority = self.current_pr["true_priority"]
        
        if pred == true_priority:
            reward = 1.0
        elif abs(pred - true_priority) == 1:
            reward = 0.5
        else:
            reward = 0.2
            if "security" in self.current_pr["labels"] and pred == 0:
                reward = 0.0
            if "critical" in self.current_pr["labels"] and pred < 2:
                reward = 0.0
        
        self.done = True
        obs = self.reset()  # return same PR's observation (new one after reset)
        info = {"true_priority": true_priority, "explanation": self._explain(pred, true_priority)}
        return obs, reward, self.done, info

    def _explain(self, pred, true):
        if pred == true:
            return "✅ Perfect match!"
        elif abs(pred - true) == 1:
            return "⚠️ Off by one – close enough."
        else:
            return "❌ Wrong priority. Check urgency/security labels."

    def state(self) -> State:
        if self.current_pr:
            obs = Observation(
                pr_title=self.current_pr["title"],
                pr_description=self.current_pr["description"],
                files_changed=self.current_pr["files_changed"],
                labels=self.current_pr["labels"],
                author=self.current_pr["author"]
            )
            return State(observation=obs, done=self.done)
        return State(observation=None, done=self.done)