# 🔨 ForgeRL — Multi-Agent Software Engineering RL Environment

<p align="center">
  <b>An OpenEnv-compatible reinforcement learning environment where LLMs learn to orchestrate
  multi-agent teams to autonomously build working software.</b>
</p>

<p align="center">
  <em>Meta PyTorch × HuggingFace OpenEnv National Hackathon Submission</em>
</p>

---

## 🎯 What is ForgeRL?

ForgeRL transforms **ForgeAI** — a production-grade multi-agent SDLC framework — into an **OpenEnv-compatible RL environment**. An LLM "meta-agent" must learn to be a **Software Engineering Manager**, deciding:

- **Which sub-agent to invoke** (Intake, Architect, Planner, QA, Coder, Recovery, Security, Oversight)
- **When to approve or reject** intermediate outputs
- **How to recover** from test failures and errors
- **When to escalate** vs. retry
- **How to adapt** to changing reviewer preferences

### Why This Environment is Novel

| Dimension | ForgeRL |
|---|---|
| **Action Space** | 17 orchestration actions across 9 sub-agents |
| **Episode Length** | 50-300+ steps (Tier 1→5 difficulty) |
| **Observation** | Partially observable — agent sees outputs, not internals |
| **Reward** | 11-component composite: dense shaping + sparse terminal |
| **Tools** | Real file system, pytest runner, Docker, LLM APIs |
| **Difficulty** | Adaptive curriculum with auto-promotion across 5 tiers |

---

## 🏆 Hackathon Theme Coverage

| Theme | How ForgeRL Addresses It |
|---|---|
| **Multi-Agent Interactions** | Meta-agent coordinates 9 specialized sub-agents with handoffs, cooperation, and partial observability |
| **Fleet AI (Scalable Oversight)** | Dedicated Oversight Agent monitors, analyzes, and explains sub-agent behavior |
| **Halluminate (Multi-Actor)** | Meta-agent manages multiple actors in a partially observable setting |
| **Long-Horizon Reasoning** | 50-300+ step episodes with sparse/delayed rewards requiring goal decomposition |
| **Professional Tasks** | Real interaction with file systems, pytest, Docker — no shortcuts |
| **Self-Improvement** | Adaptive difficulty curriculum: Tier 1 (CRUD) → Tier 5 (MongoDB joins) |
| **Snorkel AI (Expert-in-Loop)** | Simulated reviewers with changing preferences mid-episode |
| **Mercor (Token Scaling)** | Rewards scale proportionally with useful output tokens |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   ForgeRL OpenEnv Environment                 │
│                                                               │
│  ┌──────────┐   ┌───────────────────────────────────────┐    │
│  │ OpenEnv  │   │         ForgeAI Sub-Agents             │    │
│  │ HTTP API │◄─►│                                        │    │
│  │          │   │  Intake → Architect → Planner           │    │
│  │ reset()  │   │     ↓         ↓          ↓              │    │
│  │ step()   │   │    QA  →   Coder  →  Recovery           │    │
│  │ state()  │   │     ↓         ↓          ↓              │    │
│  └──────────┘   │  Security  Oversight  Reviewer          │    │
│                 └───────────────────────────────────────┘    │
│                                                               │
│  ┌────────────────────┐  ┌─────────────────────────────┐    │
│  │   Reward System     │  │   Adaptive Curriculum       │    │
│  │ 11 signal components│  │  Tier 1→5 auto-promotion    │    │
│  └────────────────────┘  └─────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/forgerl.git
cd forgerl
pip install -r forge_env/requirements.txt
pip install -r forgeai/requirements.txt
```

### Run the Demo

```bash
# Simulated mode (no API key needed)
python demo/run_demo.py --tier 1

# With real LLM sub-agents
export GOOGLE_API_KEY="your-key-here"
python demo/run_demo.py --tier 1 --real-llm
```

### Run the OpenEnv Server

```bash
uvicorn forge_env.server:app --host 0.0.0.0 --port 7860
# Visit http://localhost:7860/docs for API documentation
```

### Evaluate Policies

```bash
# Random baseline
python training/eval_forgerl.py --baseline --episodes 5

# Heuristic policy
python training/eval_forgerl.py --episodes 5 --max-tier 3
```

### Train with GRPO (Colab)

```bash
# Local training (requires GPU)
python training/train_forgerl.py --steps 500 --model unsloth/Qwen3-1.7B-Base
```

Or use the [Colab Notebook](training/ForgeRL_Training.ipynb) for cloud training.

---

## 📦 Project Structure

```
forgerl/
├── forge_env/                   # OpenEnv RL Environment
│   ├── __init__.py              # Package exports
│   ├── models.py                # Action, Observation, State models
│   ├── environment.py           # Core OpenEnv Environment
│   ├── reward.py                # 11-signal reward calculator
│   ├── curriculum.py            # Adaptive difficulty + reviewer personalities
│   ├── server.py                # FastAPI + Gradio server
│   ├── requirements.txt         # Environment dependencies
│   └── Dockerfile               # Container deployment
│
├── forgeai/                     # Multi-Agent SDLC Engine
│   ├── agents/
│   │   ├── base_agent.py        # Abstract base class
│   │   ├── intake_agent.py      # Requirements analysis
│   │   ├── architect_agent.py   # System design
│   │   ├── planner_agent.py     # Task decomposition
│   │   ├── qa_agent.py          # TDD test generation
│   │   ├── coder_agent.py       # Code generation
│   │   ├── recovery_agent.py    # Failure diagnosis
│   │   ├── security_agent.py    # Security audit
│   │   ├── oversight_agent.py   # Fleet AI oversight (NEW)
│   │   └── simulated_reviewer.py # Expert-in-loop (NEW)
│   ├── core/
│   │   ├── orchestrator.py      # Pipeline orchestration
│   │   └── activity_logger.py   # Action logging
│   ├── models/                  # Pydantic data models
│   ├── tools/                   # File manager, test runner, LLM gateway
│   └── config/                  # YAML configuration
│
├── training/                    # RL Training Scripts
│   ├── train_forgerl.py         # GRPO with Unsloth + TRL
│   ├── eval_forgerl.py          # Evaluation & reward curves
│   └── ForgeRL_Training.ipynb   # Colab notebook
│
├── demo/
│   └── run_demo.py              # Interactive demo
│
├── Dockerfile                   # Container for deployment
└── README.md                    # This file
```

---

## 🎓 Reward System

ForgeRL uses an 11-component composite reward with both dense (per-step) and sparse (terminal) signals:

| Component | Type | Value | Purpose |
|---|---|---|---|
| Phase Transition | Dense | +0.5 | Encourage forward progress |
| Task Completion | Dense | +2.0 | Reward successful task execution |
| Recovery Success | Dense | +1.0 | Incentivize graceful failure handling |
| Oversight Catch | Dense | +0.5 | Reward quality monitoring |
| Valid Delegation | Dense | +0.1 | Small reward for correct agent selection |
| Step Cost | Dense | -0.01 | Encourage efficiency |
| Invalid Action | Dense | -1.0 | Penalize invalid state transitions |
| Test Pass Rate | Terminal | ×10.0 | Scale with final test quality |
| Code Quality | Terminal | ×5.0 | Scale with oversight quality score |
| Full Success | Terminal | +20.0 | Bonus for completing all tasks |
| Token Scaling | Dense | ×0.1/1K | Mercor: scale with output tokens |

---

## 🔧 Training Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Base Model  │────►│ GRPO Trainer │────►│ Trained Model │
│  (Qwen 1.7B)│     │ (Unsloth+TRL)│     │               │
└─────────────┘     └──────┬───────┘     └───────────────┘
                           │
                    ┌──────▼───────┐
                    │   ForgeRL    │
                    │ Environment  │
                    │ (reward fn)  │
                    └──────────────┘
```

---

## 📊 Evaluation Criteria Alignment

| Criterion | Weight | Our Evidence |
|---|---|---|
| **Environment Innovation** | 40% | First RL environment for multi-agent SDLC with real tool use |
| **Storytelling** | 30% | "Can an LLM learn to be a software engineering manager?" |
| **Reward Improvement** | 20% | GRPO training curves + before/after behavior comparison |
| **Training Pipeline** | 10% | Complete Colab notebook with Unsloth + TRL GRPO |

---

## 🛠️ Tech Stack

- **Environment**: OpenEnv (latest) — Gymnasium-style `reset`/`step`/`state` API
- **Sub-Agents**: Google Gemini (Flash/Pro) via `google-generativeai`
- **Training**: HuggingFace TRL (`GRPOTrainer`) + Unsloth (4-bit quantization)
- **Server**: FastAPI + Gradio (HuggingFace Spaces)
- **Models**: Pydantic v2 for all data contracts
- **Testing**: pytest for TDD validation
- **Deployment**: Docker + HuggingFace Spaces

---

## 📄 License

MIT License — Built for the Meta PyTorch × HuggingFace OpenEnv Hackathon.
