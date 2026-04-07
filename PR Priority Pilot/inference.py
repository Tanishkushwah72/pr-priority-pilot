import os
import json
import random
import requests
import sys

# ---------- Environment variables (validator injects these) ----------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")          # ← correct name
SPACE_URL = os.environ.get("SPACE_URL", "https://tanishkushwah72-verity-human-verification.hf.space")

# Optional: MODEL_NAME may also be provided
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

# ---------- Initialize OpenAI client with proxy credentials ----------
USE_MOCK = False
try:
    from openai import OpenAI
    if API_BASE_URL and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        USE_MOCK = False
    else:
        USE_MOCK = True
        client = None
except ImportError:
    USE_MOCK = True
    client = None

def llm_priority_mock(obs):
    """Fallback when proxy credentials are missing (local testing only)."""
    text = (obs.get("pr_title", "") + " " + obs.get("pr_description", "")).lower()
    if "urgent" in text or "critical" in text or "security" in text or "hotfix" in text:
        return 2
    elif "feature" in text or "update" in text or "refactor" in text:
        return 1
    else:
        return 0

def llm_priority(obs):
    """Call the LLM through the provided proxy, or fallback to mock."""
    if USE_MOCK:
        print("⚠️ Using mock LLM (no proxy credentials)", file=sys.stderr)
        return llm_priority_mock(obs)
    try:
        prompt = f"""PR Title: {obs.get('pr_title', '')}
Description: {obs.get('pr_description', '')}
Files changed: {obs.get('files_changed', 0)}
Labels: {obs.get('labels', [])}
Author: {obs.get('author', '')}
Return only an integer 0 (Low), 1 (Medium), or 2 (High)."""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        answer = resp.choices[0].message.content.strip()
        return int(answer)
    except Exception as e:
        print(f"LLM error: {e}, using mock fallback", file=sys.stderr)
        return llm_priority_mock(obs)

def evaluate_task(task, episodes=3):
    base_url = SPACE_URL.rstrip('/')
    total_reward = 0.0
    try:
        # Reset to get a session
        reset_resp = requests.post(f"{base_url}/reset", json={"task": task}, timeout=10)
        if reset_resp.status_code != 200:
            print(f"Reset failed for {task}: {reset_resp.text}", file=sys.stderr)
            return 0.0
        session_id = reset_resp.json()["session_id"]

        for ep in range(episodes):
            # Get a new PR
            pr_resp = requests.post(f"{base_url}/reset", json={"session_id": session_id, "task": task}, timeout=10)
            if pr_resp.status_code != 200:
                print(f"PR reset failed: {pr_resp.text}", file=sys.stderr)
                continue
            obs = pr_resp.json()["observation"]
            action = llm_priority(obs)
            # Correct API call: query param + body with priority
            step_resp = requests.post(
                f"{base_url}/step?session_id={session_id}",
                json={"priority": action},
                timeout=10
            )
            if step_resp.status_code != 200:
                print(f"Step failed: {step_resp.text}", file=sys.stderr)
                continue
            step_data = step_resp.json()
            reward = step_data.get("reward", 0.0)
            total_reward += reward
            print(json.dumps({"event": "STEP", "episode": ep, "task": task, "action": action, "reward": reward}))
    except Exception as e:
        print(f"Evaluation error for {task}: {e}", file=sys.stderr)
        return 0.0
    return total_reward / episodes if episodes > 0 else 0.0

def main():
    print("[START]")
    scores = {}
    for task in ["easy", "medium", "hard"]:
        print(json.dumps({"event": "START_TASK", "task": task}))
        score = evaluate_task(task)
        scores[task] = score
        print(json.dumps({"event": "END_TASK", "task": task, "score": score}))
    print("[END]")
    print("Final scores:", json.dumps(scores))
    sys.exit(0)

if __name__ == "__main__":
    random.seed(42)
    main()
