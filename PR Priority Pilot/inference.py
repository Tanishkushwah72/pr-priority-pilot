import os
import json
import random
import requests
from openai import OpenAI

API_BASE = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
TOKEN = os.environ.get("HF_TOKEN", "")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE, api_key=TOKEN)

def llm_priority(obs):
    prompt = f"""You are a senior developer prioritizing pull requests.
PR Title: {obs['pr_title']}
Description: {obs['pr_description']}
Files changed: {obs['files_changed']}
Labels: {obs['labels']}
Author: {obs['author']}
Return only an integer 0 (Low), 1 (Medium), or 2 (High)."""
    resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
    return int(resp.choices[0].message.content.strip())

def evaluate_task(task, episodes=3):
    sess = requests.post(f"{SPACE_URL}/reset", json={"task": task}).json()["session_id"]
    total = 0
    for ep in range(episodes):
        obs = requests.post(f"{SPACE_URL}/reset", json={"session_id": sess, "task": task}).json()["observation"]
        action = llm_priority(obs)
        step = requests.post(f"{SPACE_URL}/step", json={"session_id": sess, "action": {"priority": action}}).json()
        total += step["reward"]
        print(json.dumps({"event":"STEP","episode":ep,"task":task,"action":action,"reward":step["reward"]}))
    return total / episodes

def main():
    random.seed(42)
    print("[START]")
    scores = {}
    for task in ["easy", "medium", "hard"]:
        print(json.dumps({"event":"START_TASK","task":task}))
        score = evaluate_task(task)
        scores[task] = score
        print(json.dumps({"event":"END_TASK","task":task,"score":score}))
    print("[END]")
    print("Final scores:", json.dumps(scores))

if __name__ == "__main__":
    main()