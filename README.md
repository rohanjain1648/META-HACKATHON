# ForgeAI: Agentic AI Software Development Framework

ForgeAI is an autonomous multi-agent system designed to transform natural-language project ideas into fully functional, tested, and high-quality software applications. Built for the Itlanta Hackathon 2026.

## 🚀 Key Features

*   **Intelligent Agent Orchestration:** A team of specialized agents (Intake, Architect, Planner, QA, Coder, Security, Recovery) collaborating through a robust finite state machine.
*   **TDD-First Approach:** Non-negotiable Test-Driven Development. Our QA agent writes failing tests *before* any production code is generated.
*   **Autonomous Failure Recovery:** Self-diagnosing and self-correcting agents that can recover from syntax errors, test failures, and logic gaps.
*   **Safety & Guardrails:** Configurable YAML-based guardrails that prevent unsafe operations and ensure project-level isolation.
*   **Premium CLI & Dashboard:** A high-end terminal experience with Rich animations and a real-time observability dashboard.

## 🏗️ Architecture

ForgeAI uses a layered architecture:
- **UI Layer:** CLI (Rich) and Web (React/FastAPI).
- **Orchestration Layer:** Manages agent state transitions and artifact persistence.
- **Agent Layer:** Specialized agents powered by Google Gemini (Flash/Pro).
- **Tool Layer:** File management, Test runner, Docker builder, and LLM Gateway.

## 📦 Project Complexity Tiers

ForgeAI is designed to handle all 5 hackathon tiers:
1.  **The Ledger:** Basic CRUD microservices.
2.  **Logic Engine:** Complex business rule engines.
3.  **Live Bridge:** Async 3rd-party API integrations.
4.  **The Gatekeeper:** OAuth2/JWT & RBAC.
5.  **Mongo-SQL Engine:** Real-time data syncing and complex joins.

## 🛠️ Setup & Usage

1.  Clone the repository and install dependencies: `pip install -r forgeai/requirements.txt`
2.  Set your Gemini API key in a `.env` file: `GOOGLE_API_KEY=...`
3.  Launch the framework: `python -m forgeai.main`

## ⚖️ Evaluation Alignment

| Criterion | Score | Our Strategy |
| :--- | :---: | :--- |
| Agentic Autonomy | 30 | Zero-touch from spec to code. |
| TDD & Verification | 25 | Strict TDD-first pytest generation. |
| Complex Logic | 20 | Robust state-machine and atomic task planning. |
| Failure handling | 10 | Recovery agent with diagnosis and retry cascade. |
| Code Quality | 10 | Framework is modular, Pydantic-typed, and documented. |
| Extended Features | 5 | Docker, Security Audit, and Dashboard support. |
