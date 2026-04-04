import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from environment import CodeReviewEnv, Action

app = FastAPI()
sessions = {}

class StepRequest(BaseModel):
    priority: int

@app.post("/reset")
async def reset(session_id: str = None, task: str = "easy"):
    if not session_id or session_id not in sessions:
        session_id = session_id or str(uuid.uuid4())
        sessions[session_id] = CodeReviewEnv()
    sessions[session_id].set_task(task)
    obs = sessions[session_id].reset()
    return {"session_id": session_id, "observation": obs.dict()}

@app.post("/step")
async def step(session_id: str, req: StepRequest):
    print(f"Step called: session={session_id}, priority={req.priority}")
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    action = Action(priority=req.priority)
    obs, reward, done, info = sessions[session_id].step(action)
    print(f"Returning reward={reward}")
    return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}

@app.get("/state")
async def state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"state": sessions[session_id].state().dict()}

# ---------- SIMPLE WORKING HTML ----------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PR Priority Pilot</title>
    <style>
        body { font-family: Arial; background: #1e3c72; padding: 20px; }
        .container { max-width: 800px; margin: auto; background: white; border-radius: 20px; padding: 20px; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; }
        .low { background: #48bb78; color: white; }
        .medium { background: #ed8936; color: white; }
        .high { background: #e53e3e; color: white; }
        .pr-card { background: #f0f4ff; padding: 15px; border-radius: 15px; margin: 15px 0; }
        .badge { background: #667eea; color: white; padding: 2px 8px; border-radius: 15px; font-size: 12px; margin-right: 5px; }
        .result { padding: 10px; border-radius: 10px; margin-top: 10px; }
        .good { background: #c6f6d5; color: #22543d; }
        .medium-c { background: #feebc8; color: #7b341e; }
        .bad { background: #fed7d7; color: #742a2a; }
        .score { font-size: 24px; font-weight: bold; }
        button:disabled { opacity: 0.5; }
        .loading { display: inline-block; width: 16px; height: 16px; border: 2px solid #ccc; border-top-color: #667eea; border-radius: 50%; animation: spin 0.6s linear infinite; margin-left: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <h1>🚀 PR Priority Pilot</h1>
    <p>Click a priority button to earn points</p>

    <div>
        <strong>Difficulty:</strong>
        <label><input type="radio" name="task" value="easy" checked> Easy</label>
        <label><input type="radio" name="task" value="medium"> Medium</label>
        <label><input type="radio" name="task" value="hard"> Hard</label>
        <button id="resetBtn">⟳ Reset</button>
    </div>

    <div class="pr-card" id="prCard">Click Reset to start</div>

    <div>
        <button class="low" data-priority="0">🐞 Low</button>
        <button class="medium" data-priority="1">⚙️ Medium</button>
        <button class="high" data-priority="2">🔥 High</button>
    </div>

    <div id="resultArea" style="display:none;" class="result"></div>

    <h3>🏆 Total Score: <span id="totalScore">0.0</span></h3>
    <div>PRs reviewed: <span id="prCount">0</span></div>
    <div id="history"></div>
    <div id="debug" style="margin-top:20px; border-top:1px solid #ccc; font-size:12px; color:gray;"></div>
</div>

<script>
    let sessionId = null;
    let currentTask = "easy";
    let totalScore = 0.0;
    let reviewedCount = 0;
    let history = [];

    function debug(msg) {
        console.log(msg);
        const debugDiv = document.getElementById("debug");
        if (debugDiv) debugDiv.innerHTML += new Date().toLocaleTimeString() + " " + msg + "<br>";
    }

    function updateUI() {
        document.getElementById("totalScore").innerText = totalScore.toFixed(1);
        document.getElementById("prCount").innerText = reviewedCount;
        let historyHtml = "<strong>History:</strong><br>";
        history.slice(-5).reverse().forEach(h => { historyHtml += h + "<br>"; });
        document.getElementById("history").innerHTML = historyHtml;
    }

    async function apiCall(endpoint, method, body) {
        const url = window.location.origin + endpoint;
        debug(`Calling ${method} ${url} ${body ? JSON.stringify(body) : ""}`);
        const options = { method, headers: { "Content-Type": "application/json" } };
        if (body) options.body = JSON.stringify(body);
        const res = await fetch(url, options);
        if (!res.ok) {
            const err = await res.text();
            debug(`Error: ${err}`);
            throw new Error(err);
        }
        const data = await res.json();
        debug(`Response: ${JSON.stringify(data)}`);
        return data;
    }

    async function loadPR(resetScore = false) {
        debug(`loadPR resetScore=${resetScore}, task=${currentTask}`);
        document.getElementById("prCard").innerHTML = "Loading... <span class='loading'></span>";
        try {
            const data = await apiCall("/reset", "POST", { task: currentTask });
            sessionId = data.session_id;
            renderPR(data.observation);
            if (resetScore) {
                totalScore = 0.0;
                reviewedCount = 0;
                history = [];
                updateUI();
                debug("Score reset");
            }
        } catch (err) {
            document.getElementById("prCard").innerHTML = "Error: " + err.message;
            alert("Reset failed: " + err.message);
        }
    }

    function renderPR(obs) {
        const labels = (obs.labels || []).map(l => `<span class="badge">${escapeHtml(l)}</span>`).join('');
        document.getElementById("prCard").innerHTML = `
            <h3>${escapeHtml(obs.pr_title)}</h3>
            <p>${escapeHtml(obs.pr_description)}</p>
            <div>📄 ${obs.files_changed} files | 👤 ${escapeHtml(obs.author)}</div>
            <div>${labels}</div>
        `;
    }

    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    async function takeAction(priority, priorityName) {
        if (!sessionId) {
            alert("Please reset first.");
            return;
        }
        debug(`takeAction priority=${priority} name=${priorityName}`);
        const resultDiv = document.getElementById("resultArea");
        resultDiv.style.display = "block";
        resultDiv.innerHTML = "Processing... <span class='loading'></span>";
        try {
            const stepData = await apiCall(`/step?session_id=${sessionId}`, "POST", { priority: priority });
            const reward = stepData.reward;
            const explanation = stepData.info?.explanation || "";
            totalScore += reward;
            reviewedCount++;
            const timestamp = new Date().toLocaleTimeString();
            const icon = reward === 1.0 ? "✅" : (reward === 0.5 ? "⚠️" : "❌");
            history.unshift(`${timestamp} ${icon} ${priorityName} → ${reward} (${explanation})`);
            if (history.length > 10) history.pop();
            updateUI();
            debug(`Reward ${reward}, new total ${totalScore}`);
            let rewardClass = "";
            let rewardText = "";
            if (reward === 1.0) { rewardText = "🏆 Perfect! +1.0"; rewardClass = "good"; }
            else if (reward === 0.5) { rewardText = "⚠️ Close enough! +0.5"; rewardClass = "medium-c"; }
            else { rewardText = "❌ Wrong priority! +0.0"; rewardClass = "bad"; }
            resultDiv.innerHTML = `<strong>${rewardText}</strong><br>Reward: ${reward}<br>${explanation}`;
            resultDiv.className = `result ${rewardClass}`;
            // Load next PR after 1 second
            setTimeout(() => {
                loadPR(false);
                resultDiv.style.display = "none";
            }, 1000);
        } catch (err) {
            debug(`Action error: ${err.message}`);
            resultDiv.innerHTML = `<span style="color:red;">Error: ${err.message}</span>`;
            resultDiv.className = "result bad";
            alert("Action failed: " + err.message);
        }
    }

    // Event listeners
    document.querySelectorAll('input[name="task"]').forEach(radio => {
        radio.addEventListener("change", async (e) => {
            if (e.target.checked) {
                currentTask = e.target.value;
                await loadPR(true);
            }
        });
    });
    document.getElementById("resetBtn").addEventListener("click", () => loadPR(true));
    document.querySelectorAll(".low, .medium, .high").forEach(btn => {
        btn.addEventListener("click", () => {
            const priority = parseInt(btn.getAttribute("data-priority"));
            const name = btn.innerText.trim();
            takeAction(priority, name);
        });
    });

    debug("Page loaded. Starting initial reset.");
    loadPR(true);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()