# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r forgeai/requirements.txt

# Run the framework (interactive prompt if --spec omitted)
python -m forgeai.main --spec "Build a simple Task Management API"

# Run with a custom YAML config
python -m forgeai.main --spec "..." --config path/to/config.yaml

# Launch web dashboard mode
python -m forgeai.main --web

# Generate architecture diagrams (requires: pip install graphviz)
python generate_diagrams.py

# Run tests
pytest
```

## Architecture

ForgeAI is a multi-agent SDLC automation framework. A natural-language spec flows through a **16-state finite state machine** (FSM) and produces a tested, generated project.

### Four-layer stack

| Layer | Location | Role |
|---|---|---|
| UI | `forgeai/ui/` | Rich CLI (`cli_interface.py`) and FastAPI web dashboard (`web_server.py`) |
| Orchestration | `forgeai/core/` | `Orchestrator` (FSM engine), `ActivityLogger`, `ConfigManager` |
| Agents | `forgeai/agents/` | 7 specialized agents, all inheriting `BaseAgent` |
| Tools | `forgeai/tools/` | `LLMGateway`, `FileManager`, `TestRunner`, `DockerBuilder` |

### Pipeline phases (FSM)

```
IDLE → INTAKE → CLARIFICATION → SPECIFICATION → ARCHITECTURE → PLANNING
     → PLAN_REVIEW (human checkpoint) → EXECUTION → SECURITY_AUDIT → SUMMARY → DONE
```

Within `EXECUTION`, each task runs a TDD inner loop:
```
TASK_QA (write failing tests) → TASK_CODE (write code to pass) → TASK_TEST (run pytest)
                                     ↑                                        |
                                     └──────── TASK_RECOVERY ←────────────────┘
```

### Agent contract

Every agent is a subclass of `BaseAgent` (`forgeai/agents/base_agent.py`) and implements three abstract methods:
- `build_system_prompt()` — static system instruction for the LLM
- `build_user_prompt(context: AgentContext)` — task-specific prompt derived from context
- `parse_response(raw_response, context)` — converts the LLM's text into a typed `AgentResult`

Agents never call the LLM directly — they go through `LLMGateway`, which handles exponential-backoff retries, token counting, and JSON extraction.

### Data flow

`AgentContext` (input) → `BaseAgent.execute()` → LLM → `AgentResult` (output)

Artifacts are accumulated in `WorkflowState` and written to `.forgeai/` inside the generated project directory (`./generated_project` by default).

### Recovery cascade

When tests fail, `RecoveryAgent` classifies the error and returns one of four strategies: `RETRY_WITH_FIX`, `MODIFY_APPROACH`, `SKIP_TASK`, or `ESCALATE`. The orchestrator respects `workflow.max_retries` (default 3) before giving up on a task.

### Configuration

All behaviour is controlled by `forgeai/config/default_config.yaml` (or a custom YAML passed via `--config`). Key knobs:
- `workflow.auto_approve_checkpoints: true` — skip all human approval prompts (useful for CI/automated runs)
- `workflow.max_retries` — recovery retry limit per task
- `security_audit.enabled` — toggle the post-execution security scan
- `output.project_dir` — where generated files land

The `ConfigManager` is a singleton; call `ConfigManager.reset()` in tests to get a fresh instance.

### LLM provider

Currently hard-wired to Google Gemini via `google-generativeai`. The `LLMGateway` is provider-agnostic in interface but only implements the `"google"` branch. The API key is read from `GOOGLE_API_KEY` (set in `.env`).
