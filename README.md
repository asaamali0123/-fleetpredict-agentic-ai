# FleetPredict AI — Agentic Fleet Diagnostic & Predictive Maintenance Copilot

## 1. Overview

FleetPredict AI combines deterministic fleet telemetry analytics with a real multi-agent CrewAI reasoning layer and evidence-grounded maintenance RAG.

**Core principle:** Python calculates facts. Agents interpret them. RAG supplies maintenance evidence.

## 2. Architecture

```text
CSV
 │
 ▼
Pandas Analytics ──► anomaly detection ──► health score ──► risk ranking
 │
 ▼
CrewAI Sequential Crew
 │
 ├── Fleet Data Analyst
 ├── Vehicle Health & Risk Agent
 ├── Root Cause Diagnosis Agent
 ├── Maintenance Knowledge / RAG Agent
 ├── Maintenance Planner Agent
 └── Fleet Manager / Final Verification Agent
                         ▲
                         │
                 FAISS + SentenceTransformers
                         │
                  maintenance evidence
                         │
                         ▼
                 Evidence → Diagnosis → Recommendation
```

## 3. Agents

### Agent 1 — Fleet Data Analyst
Interprets deterministic fleet statistics, anomalies and patterns.

### Agent 2 — Vehicle Health & Risk Agent
Explains deterministic health scores and risk categories.

### Agent 3 — Root Cause Diagnosis Agent
Connects sensor combinations into cautious hypotheses and explicitly separates anomaly from likely/possible cause.

### Agent 4 — Maintenance Knowledge / RAG Agent
Uses retrieved FAISS/SentenceTransformer chunks as factual maintenance evidence.

### Agent 5 — Maintenance Planner Agent
Converts evidence + diagnosis into prioritized maintenance actions.

### Agent 6 — Fleet Manager / Final Decision Agent
Acts as the verification gate and produces the executive report.

## 4. Deterministic rules

- Engine temperature > 105 → -20
- Oil pressure < 30 → -25
- Vibration > 7 → -20
- Battery voltage < 11.8 → -15

Health:

- 90–100 Healthy
- 75–89 Watch
- 50–74 Warning
- 0–49 Critical

Missing optional sensor columns are tolerated. Column aliases include `engine_temperature`, `oil_press`, `battery`, `vibration_level`, etc.

## 5. Features

- Fleet risk ranking
- Vehicle diagnostic report
- Six-agent CrewAI orchestration
- Groq inference
- FAISS + SentenceTransformers RAG
- Evidence panel
- Evidence → Diagnosis → Recommendation verification
- Automatic maintenance priority
- AI executive report
- What-If simulator
- Graceful AI failure

## 6. Local setup

Recommended: Python 3.11/3.12.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Optional local `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY="YOUR_REAL_KEY"
GROQ_MODEL="openai/gpt-oss-120b"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K="3"
```

## 7. RAG setup

The app includes a demo knowledge corpus and can repair/rebuild the demo FAISS store automatically.

For a real manual:

```text
knowledge_base/manuals/your_manual.pdf
```

Then:

```bash
python scripts/build_vector_store.py
```

Commit the generated:

```text
knowledge_base/vector_store/index.faiss
knowledge_base/vector_store/metadata.json
```

Runtime loads the index and embedding model through Streamlit resource caching and embeds only the user query.

## 8. GitHub — exact steps

From the extracted project folder:

```bash
git init
git add .
git commit -m "FleetPredict AI hackathon MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fleetpredict-ai.git
git push -u origin main
```

Do not commit API keys.

## 9. Streamlit Community Cloud — exact steps

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app**.
4. Select your repository.
5. Branch: `main`.
6. Main file: `app.py`.
7. Open **Advanced settings → Secrets**.
8. Paste:

```toml
GROQ_API_KEY="YOUR_REAL_GROQ_KEY"
GROQ_MODEL="openai/gpt-oss-120b"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K="3"
```

9. Deploy.
10. Wait for dependency installation.
11. Open the app URL.

## 10. First deployment test

Use the included:

```text
data/sample_telemetry.csv
```

In the app:

1. Keep **Use demo telemetry** checked.
2. Select `TRK-002`.
3. Confirm health is Critical.
4. Click **Run Agentic Diagnosis**.
5. Check all six agent tabs.
6. Open Retrieved Maintenance Evidence.
7. Verify Evidence → Diagnosis → Recommendation.
8. Change a What-If value.

## 11. 60–90 second demo

> “FleetPredict AI is an agentic predictive-maintenance copilot. We deliberately separate deterministic facts from AI reasoning. Pandas handles telemetry normalization, threshold anomalies, health scoring and fleet risk ranking. CrewAI then orchestrates six specialists. The analyst interprets fleet patterns, the health agent evaluates risk, the root-cause agent creates cautious hypotheses, the RAG agent grounds those hypotheses in maintenance evidence, the planner creates prioritized actions, and the Fleet Manager verifies the final evidence-to-recommendation chain.
>
> Here is TRK-002: it has high temperature, low oil pressure and excessive vibration. The system does not claim a confirmed mechanical failure. Instead, it identifies possible causes and retrieves maintenance evidence. The planner converts that into an actionable inspection priority.
>
> Finally, the What-If simulator lets an operator test sensor changes instantly without an LLM call, and the Fleet Manager produces an executive fleet report.”

## 12. Troubleshooting

### AI unavailable
Add `GROQ_API_KEY` under Streamlit Secrets and redeploy/reboot.

### Model error
Change `GROQ_MODEL` to another currently supported Groq model. The default is `openai/gpt-oss-120b`.

### FAISS startup issue
The app automatically attempts to rebuild the demo index. If needed, delete `knowledge_base/vector_store/index.faiss` and restart.

### Real PDF
```bash
python scripts/build_vector_store.py
```

### Dependency issue
Use Python 3.12 and redeploy.

### Deterministic dashboard still works
If Groq/CrewAI fails, the dashboard continues to provide analytics, anomalies, health, ranking and What-If results.

## 13. Security

Never put the real Groq key in Python, GitHub, `.env` committed to Git, or README. Use Streamlit Secrets.

## 14. Future improvements

Time-series forecasting, work-order integration, authenticated roles, OEM-specific manuals, failure probability models, maintenance history and human approval workflows.
