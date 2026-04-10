# ForgeAI — Detailed Design Notes
## Phase 1 Design Document | Itlanta Hackathon 2026

---

# TABLE OF CONTENTS

1. [Agent Architecture](#1-agent-architecture)
2. [Workflow Design](#2-workflow-design)
3. [Failure Strategy](#3-failure-strategy)
4. [Tech Stack Justification](#4-tech-stack-justification)
5. [Risk Assessment](#5-risk-assessment)

---

# 1. AGENT ARCHITECTURE

## 1.1 Overview — Why Multi-Agent?

ForgeAI uses a **7-agent collaborative architecture** where each agent is a specialized, isolated unit responsible for exactly one phase of the software development lifecycle. This design follows the **Single Responsibility Principle** — no agent does more than one job, and no two agents share the same responsibility.

The fundamental insight is: **a single monolithic LLM prompt cannot handle the full complexity of software development**. By decomposing the problem into specialized agents, each agent can have a focused system prompt, receive only the context it needs, and produce a well-defined output that the next agent consumes.

## 1.2 The 7 Agents — Deep Dive

### Agent 1: Intake Agent (`intake_agent.py`)

**Purpose:** The Intake Agent is the first point of contact. It receives a raw, natural-language project specification from the user and transforms it into a machine-readable, structured specification.

**Responsibilities:**
- Parse and analyze the natural-language input to understand what the user wants to build
- **Detect ambiguities** — identify parts of the spec that are vague, underspecified, or contradictory (e.g., "build a REST API" doesn't specify which endpoints, data models, auth requirements, etc.)
- **Generate targeted clarifying questions** — produce 5-7 specific questions to eliminate ambiguity (e.g., "Should the API require authentication? If yes, which method — JWT, OAuth2, or API keys?")
- After receiving answers, produce a **StructuredSpecification** — a Pydantic model containing:
  - `project_name` — descriptive name for the project
  - `summary` — 1-2 sentence description
  - `tier` — complexity tier (1-5)
  - `acceptance_criteria` — testable list of what "done" means
  - `constraints` — technical and business constraints
  - `functional_requirements` — detailed FR list
  - `non_functional_requirements` — NFR list
  - `tech_stack` — backend (Python), database, frontend choices
  - `data_models` — structured schema definitions with field types and validations
  - `api_endpoints` — method, path, description, auth requirements
  - `architecture_style` — monolith, microservice, or layered

**Why it matters:** The StructuredSpecification is the **single source of truth** consumed by ALL downstream agents. If the spec is wrong, everything downstream is wrong. That's why we invest heavily in clarification before moving forward.

**Satisfies:** FR-01 (accept NL spec, identify ambiguities), FR-02 (produce structured spec)

---

### Agent 2: Architect Agent (`architect_agent.py`)

**Purpose:** The Architect Agent takes the StructuredSpecification and designs the complete project structure from scratch — before any code is written.

**Responsibilities:**
- **Design directory layout** — determine the folder structure (e.g., `src/`, `tests/`, `models/`, `routes/`, `config/`)
- **Define data models** — expand on the spec's data models with exact Python class definitions, relationships, and validation rules
- **Design API contracts** — define request/response schemas, status codes, error formats
- **Make technology decisions** — choose specific libraries (e.g., FastAPI vs Flask, SQLAlchemy vs Tortoise ORM, PostgreSQL vs SQLite)
- **Output** — a structured architecture dictionary containing all design decisions

**Why it matters:** Without upfront architecture, the Coder Agent would make inconsistent decisions across tasks — one task might use Flask, another might use FastAPI. The Architect Agent ensures consistency.

**Satisfies:** FR-04 (design project structure from scratch — directory layout, modules, data models, API contracts)

---

### Agent 3: Planner Agent (`planner_agent.py`)

**Purpose:** The Planner Agent converts the specification + architecture into an ordered list of atomic implementation tasks.

**Responsibilities:**
- **Decompose** the project into 8-15 atomic tasks (each produces ONE verifiable, independently testable unit of work)
- **Build a dependency graph** — Task 3 depends on Task 1 and 2 being complete. The Planner determines this ordering.
- **Assign risk levels** — each task gets a risk level: LOW, MEDIUM, HIGH, or CRITICAL. High-risk tasks may receive extra retries.
- **Set checkpoint flags** — critical tasks (e.g., database schema, auth setup) can be flagged as checkpoints requiring human approval after completion.
- **Estimate file scope** — how many files will each task produce/modify

**Output:** An `ImplementationPlan` Pydantic model containing:
- `project_name` — name of the project
- `tasks` — ordered list of `AtomicTask` objects
- `total_estimated_files` — total file count estimate
- `architecture_summary` — brief summary of the architecture

Each `AtomicTask` contains:
- `id` — sequential integer
- `title` — short description
- `description` — detailed description of what to implement
- `target_files` — list of files to create/modify
- `dependencies` — list of task IDs that must complete first
- `risk_level` — LOW, MEDIUM, HIGH, CRITICAL
- `is_checkpoint` — whether to pause for human approval

**Why it matters:** Atomic task decomposition is critical because:
1. It enables **per-task TDD** — tests are written and validated for each task independently
2. It enables **granular failure recovery** — if Task 5 fails, Tasks 1-4 are still intact
3. It enables **dependency-aware scheduling** — tasks are executed in the right order

**Satisfies:** FR-05 (decompose into ordered atomic tasks), FR-06 (present plan for review)

---

### Agent 4: QA Agent (`qa_agent.py`) — TDD-First

**Purpose:** The QA Agent writes failing test cases BEFORE any production code is generated. This is the heart of our TDD-first approach.

**Responsibilities:**
- For each atomic task, generate **pytest test files** that define the expected behavior
- Tests must be **specific enough** to validate correctness but **flexible enough** to not over-constrain the Coder Agent
- Cover **edge cases** — empty inputs, invalid data, boundary conditions
- Cover **happy paths** — normal expected behavior
- Generate at least 3-5 test functions per task
- Tests are written in pytest format with clear assertion messages

**How it works:**
1. The Orchestrator invokes the QA Agent with the current task, specification, architecture, and existing project files
2. The QA Agent generates test files (e.g., `tests/test_user_model.py`)
3. Tests are written to disk BEFORE the Coder Agent is invoked
4. The Coder Agent can READ these test files and knows exactly what behavior is expected

**Why TDD-first is non-negotiable:**
- It gives the Coder Agent a **concrete contract** — not just a vague description, but actual test assertions to satisfy
- It provides an **automated verification mechanism** — we can automatically check if generated code is correct
- It gives the Recovery Agent **precise error signals** — test failure output tells exactly WHAT went wrong
- The hackathon scoring allocates **25 points** (out of 100) to TDD & Verification

**Satisfies:** FR-11 (TDD-first: QA generates tests before code), FR-12 (auto-run tests after code generation)

---

### Agent 5: Coder Agent (`coder_agent.py`)

**Purpose:** The Coder Agent generates production code that passes the failing tests written by the QA Agent.

**Responsibilities:**
- Receive the task description, specification, architecture, existing project files, AND the test files written by QA
- Generate production Python code that satisfies the tests
- Follow the architecture design decisions (use the right framework, follow the directory structure)
- Maintain consistency with existing project code
- On retry: receive error messages, tracebacks, and fix instructions from the Recovery Agent

**Context provided to the Coder Agent:**
- `specification` — what the project should do
- `architecture` — how the project is structured
- `current_task` — what specifically to implement now
- `existing_files` — all files already generated (so it can import and build on them)
- `error_message` — (on retry) what went wrong last time
- `previous_attempts` — (on retry) full error history
- `retry_count` — which attempt this is

**Output:** A dictionary of `{filepath → code_content}` that the Orchestrator writes to disk through the File Manager.

**Key design decision:** The Coder Agent **does NOT write files directly** — it returns file contents to the Orchestrator, which writes them through the sandboxed FileManager. This ensures safety and traceability.

**Satisfies:** FR-08 (invoke LLM to generate code), FR-09 (present changes as diff/summary)

---

### Agent 6: Security Agent (`security_agent.py`)

**Purpose:** After all tasks are complete, the Security Agent performs a comprehensive security audit on the entire generated codebase.

**Responsibilities:**
- Scan for **injection vulnerabilities** — SQL injection, command injection, XSS
- Check for **authentication flaws** — missing auth checks, weak token validation
- Detect **hardcoded secrets** — API keys, passwords, tokens in source code
- Identify **path traversal** vulnerabilities
- Check for **authorization bypass** — missing RBAC checks
- Produce a structured `security_report.json`

**Scan patterns (configurable via YAML):**
```yaml
security_audit:
  enabled: true
  scan_patterns:
    - "hardcoded_secrets"
    - "sql_injection"
    - "command_injection"
    - "path_traversal"
    - "auth_bypass"
```

**Satisfies:** FR-14 (Extended — AI-powered security audit after each major module)

---

### Agent 7: Recovery Agent (`recovery_agent.py`)

**Purpose:** When tests fail, the Recovery Agent diagnoses the root cause and recommends a recovery strategy.

**Responsibilities:**
- Receive the full error context: error message, traceback, previous attempts, current code
- **Classify the error type:** syntax, import, logic, runtime, timeout, or test_design
- **Identify where the bug is:** in the test code, production code, both, or configuration
- **Choose a recovery strategy:** RETRY_WITH_FIX, MODIFY_APPROACH, SKIP_TASK, or ESCALATE
- Provide **specific fix instructions** that are passed to the Coder Agent on retry
- Optionally **modify test code** if the bug is in the test, not the production code

**Output structure:**
```json
{
  "diagnosis": {
    "root_cause": "ImportError: module 'fastapi' has no attribute 'APIRoute'",
    "error_type": "import",
    "error_in": "production_code"
  },
  "strategy": "RETRY_WITH_FIX",
  "fix_instructions": "Replace 'from fastapi import APIRoute' with 'from fastapi import APIRouter'. The class is APIRouter, not APIRoute.",
  "modified_test_code": {},
  "confidence": 0.95
}
```

**Satisfies:** FR-15 (auto-retry with error context), FR-17 (Extended — rollback support)

---

## 1.3 Agent Isolation Principles

1. **No shared mutable state** — Agents communicate ONLY through `AgentContext` (input) and `AgentResult` (output). Both are Pydantic models with strict type validation.

2. **No filesystem access** — Agents do NOT read or write files directly. They return `{path → content}` dictionaries, and the Orchestrator writes through the sandboxed `FileManager`.

3. **No cross-agent calls** — Agent A never invokes Agent B. Only the Orchestrator invokes agents, in the correct sequence.

4. **Uniform interface** — Every agent extends `BaseAgent` and implements exactly 3 abstract methods:
   - `build_system_prompt()` → the LLM system instruction (agent's persona)
   - `build_user_prompt(context)` → the task-specific prompt built from AgentContext
   - `parse_response(raw, context)` → parse the LLM's raw text response into an AgentResult

5. **Extensibility** — Adding a new agent (e.g., Documentation Agent, Performance Agent) requires implementing just these 3 methods and registering it in the Orchestrator.

---

# 2. WORKFLOW DESIGN

## 2.1 Overview — Finite State Machine

The entire ForgeAI pipeline is governed by a **16-state Finite State Machine (FSM)** implemented in `workflow_state.py`. Every state transition is **validated** — the system checks a `VALID_TRANSITIONS` dictionary before moving to any new state. Invalid transitions are rejected and logged as errors.

This is critical because:
- It prevents the system from skipping steps (e.g., jumping from INTAKE to EXECUTION)
- It ensures the TDD loop (QA → CODE → TEST → RECOVERY) is followed correctly
- It provides a clear audit trail of what happened and when

## 2.2 The 16 States

| State | Description | What Happens |
|-------|-------------|--------------|
| `IDLE` | Initial state | System waiting for user input |
| `INTAKE` | Processing raw spec | Intake Agent analyzes NL specification |
| `CLARIFICATION` | Asking questions | Intake Agent detected ambiguities → questions sent to user |
| `SPECIFICATION` | Producing structured spec | Intake Agent generates StructuredSpecification from answers |
| `ARCHITECTURE` | Designing project | Architect Agent designs directory layout, models, API contracts |
| `PLANNING` | Decomposing tasks | Planner Agent creates ordered AtomicTask list |
| `PLAN_REVIEW` | **CHECKPOINT** | Execution pauses. User must review and approve the implementation plan |
| `EXECUTION` | TDD loop "outer" | Orchestrator iterates through tasks. Picks next task from plan |
| `TASK_QA` | Writing tests | QA Agent writes failing tests for current task |
| `TASK_CODE` | Generating code | Coder Agent generates production code to pass tests |
| `TASK_TEST` | Running tests | Test Runner executes pytest suite |
| `TASK_RECOVERY` | Handling failure | Recovery Agent diagnoses failure and recommends strategy |
| `SECURITY_AUDIT` | Scanning code | Security Agent audits entire codebase for vulnerabilities |
| `SUMMARY` | Generating report | Orchestrator produces workflow_summary.json |
| `DONE` | Pipeline complete | All artifacts generated. Summary displayed to user |
| `ERROR` | Unrecoverable error | Pipeline encountered a fatal error |

## 2.3 Valid State Transitions

```
IDLE           →  INTAKE
INTAKE         →  CLARIFICATION, SPECIFICATION
CLARIFICATION  →  SPECIFICATION, INTAKE
SPECIFICATION  →  ARCHITECTURE
ARCHITECTURE   →  PLANNING
PLANNING       →  PLAN_REVIEW
PLAN_REVIEW    →  EXECUTION, PLANNING (if revisions needed)
EXECUTION      →  TASK_QA, SECURITY_AUDIT, SUMMARY
TASK_QA        →  TASK_CODE, TASK_RECOVERY
TASK_CODE      →  TASK_TEST, TASK_RECOVERY
TASK_TEST      →  EXECUTION (pass), TASK_RECOVERY (fail)
TASK_RECOVERY  →  TASK_QA, TASK_CODE, EXECUTION, ERROR
SECURITY_AUDIT →  SUMMARY
SUMMARY        →  DONE
DONE           →  (terminal)
ERROR          →  IDLE (reset)
```

**Key design point:** Every transition is checked by `WorkflowState.transition_to()`:
```python
def transition_to(self, new_phase: WorkflowPhase) -> bool:
    valid = VALID_TRANSITIONS.get(self.phase, [])
    if new_phase in valid:
        self.phase = new_phase
        return True
    return False  # Invalid transition — rejected
```

## 2.4 How Agents Hand Off Work

Agents **do NOT** communicate directly with each other. All communication flows through the Orchestrator:

**Flow:**
```
User → Orchestrator → Intake Agent → [StructuredSpec] → Orchestrator
                    → Architect Agent → [Architecture dict] → Orchestrator
                    → Planner Agent → [ImplementationPlan] → Orchestrator
                    → [For each task]:
                        → QA Agent → [Test files] → Orchestrator → Write to disk
                        → Coder Agent → [Production files] → Orchestrator → Write to disk
                        → Test Runner → [Pass/Fail] → Orchestrator
                        → [If fail]: Recovery Agent → [Fix instructions] → Orchestrator → Coder Agent (retry)
                    → Security Agent → [Security Report] → Orchestrator
                    → Orchestrator → [Summary Report] → User
```

**What each agent produces and who consumes it:**

| Agent | Produces | Consumed By |
|-------|----------|-------------|
| Intake | StructuredSpecification | Architect, Planner, QA, Coder, Security (all agents get the spec) |
| Architect | Architecture dict (layout, models, APIs) | Planner, QA, Coder |
| Planner | ImplementationPlan (ordered task list) | Orchestrator (drives execution loop) |
| QA | Test files `{path → content}` | Written to disk. Coder reads them. Test Runner executes them. |
| Coder | Production files `{path → content}` | Written to disk. Test Runner validates them. |
| Recovery | Fix instructions + strategy | Coder Agent receives on retry. Orchestrator decides flow. |
| Security | Security report dict | Written as artifact. Included in summary. |

## 2.5 Human-in-the-Loop Checkpoints

ForgeAI has **4 configurable checkpoint types** where the pipeline pauses for human review:

### Checkpoint 1: After Specification
- **When:** After the Intake Agent produces the structured specification
- **What the user reviews:** Project name, summary, requirements, data models, API endpoints, tech stack
- **User actions:** Approve to proceed, or request changes
- **Configurable:** Yes, via `workflow.checkpoints` in YAML

### Checkpoint 2: After Architecture
- **When:** After the Architect Agent designs the project structure
- **What the user reviews:** Directory layout, technology choices, model schemas, API contracts
- **User actions:** Approve or request redesign
- **Configurable:** Yes, via `workflow.checkpoints`

### Checkpoint 3: After Plan (MANDATORY — FR-06)
- **When:** After the Planner Agent creates the implementation plan
- **What the user reviews:** Ordered task list with descriptions, risk levels, dependencies, checkpoint flags
- **Why this is mandatory:** FR-06 explicitly requires "The implementation plan must be presented to the user for review and approval before execution begins"
- **User actions:** Approve to start execution, or request plan modifications

### Checkpoint 4: Per-Diff Review (FR-09)
- **When:** Each time the Coder Agent generates code for a task
- **What the user reviews:** The generated files as a diff summary
- **User actions:** Approve the code changes, or reject (triggering a re-generation)

### Auto-Approve Mode
All checkpoints can be bypassed by setting:
```yaml
workflow:
  auto_approve_checkpoints: true
```
This enables a **fully autonomous, zero-touch demo mode** where ForgeAI runs from spec to deliverable without any human intervention — ideal for live demonstrations.

## 2.6 The TDD Execution Loop (Most Important Part)

The TDD execution loop is the core build mechanism. For EACH atomic task in the plan:

**Step 1: QA Agent writes failing tests**
- The QA Agent receives: spec, architecture, task description, and all existing project files
- It generates pytest test files that define the expected behavior
- Tests are written to disk BEFORE any production code exists
- At this point, running `pytest` would show all tests FAILING (because the production code doesn't exist yet)

**Step 2: Coder Agent generates production code**
- The Coder Agent receives: everything the QA Agent got, PLUS the test files
- It reads the tests to understand exactly what behavior is expected
- It generates production code designed to pass those tests
- The Orchestrator optionally presents the code as a diff for review (FR-09)
- Code is written to disk

**Step 3: Test Runner executes pytest**
- The Test Runner runs the full pytest suite (all tests, not just the current task)
- Returns: number passed, number failed, error output

**Step 4: Decision point**
- **If all tests pass:** Task is marked PASSED. Move to the next task.
- **If tests fail:** Trigger the Recovery Agent (Step 5)

**Step 5: Recovery Agent diagnoses** (only on failure)
- Recovery Agent receives: full error output, traceback, all previous attempts
- It diagnoses the root cause and chooses a strategy
- If RETRY_WITH_FIX: Coder Agent is re-invoked with fix instructions + error context
- If SKIP_TASK: Task is skipped, pipeline continues
- If ESCALATE: Pipeline pauses for human intervention

**Retry loop:** Steps 2-5 repeat up to `max_retries` times (configurable, default 3). Each retry gives the Coder Agent MORE context:
- Attempt 1: spec + architecture + task + tests
- Attempt 2: + error message + traceback from attempt 1
- Attempt 3: + all previous errors + Recovery Agent's fix instructions
- Attempt 4 (if configured): + complete error history → skip or escalate if still failing

## 2.7 Dependency-Aware Task Scheduling

The `ImplementationPlan.get_next_task()` method implements dependency-aware scheduling:

```python
def get_next_task(self) -> Optional[AtomicTask]:
    completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.PASSED}
    for task in self.tasks:
        if task.status == TaskStatus.PENDING:
            if all(dep in completed_ids for dep in task.dependencies):
                return task
    return None
```

This ensures:
- Tasks are only started when ALL their dependencies have PASSED
- If Task 3 depends on Task 1 and Task 2, it won't start until both are done
- If Task 1 FAILS, Task 3 (which depends on it) is automatically blocked
- Independent tasks (no dependencies on the failed task) can still proceed

---

# 3. FAILURE STRATEGY

## 3.1 Design Philosophy

ForgeAI's failure strategy is built on three principles:

1. **Failures are expected, not exceptional.** LLMs hallucinate. Generated code has bugs. This is normal. The system is designed to handle failures gracefully, not crash.

2. **Try cheapest fix first, escalate gradually.** Don't ask the human for help when a simple retry with error context would fix it. Save human intervention for truly complex problems.

3. **Every failure enriches the next attempt.** Error messages, tracebacks, and Recovery Agent diagnosis are accumulated and passed to the Coder Agent on retry, giving it progressively richer context.

## 3.2 Failure Detection Mechanisms

### Mechanism 1: Test Failures (Primary)
- The Test Runner executes pytest after every code generation step
- Any test failure is a **blocking event** (FR-12)
- The test output (stdout + stderr) provides precise error signals

### Mechanism 2: Agent Execution Failures
- If an agent throws an exception (LLM API error, JSON parse error, etc.), the `BaseAgent.execute()` method catches it and returns a failed `AgentResult`
- The Orchestrator checks `result.success` after every agent invocation

### Mechanism 3: Invalid State Transitions
- The FSM validates every transition
- If the Orchestrator attempts an invalid transition (indicating a logic bug), it's logged as an error

### Mechanism 4: LLM API Failures
- The `LLMGateway` handles API-level failures with automatic retry + exponential backoff
- Rate limiting (429 errors) → wait 2s → 4s → 8s
- Network errors → retry up to 3 times
- If all retries fail → raise RuntimeError, caught by the agent's error handler

## 3.3 The 4-Tier Recovery Cascade

When a test fails, the Recovery Agent diagnoses the issue and selects one of four strategies:

### Tier 1: RETRY_WITH_FIX (Most Common)
- **When used:** The error is clear and fixable — syntax errors, import errors, wrong function names, missing parameters
- **What happens:** Recovery Agent provides specific fix instructions. Coder Agent is re-invoked with:
  - The original context (spec, arch, task, tests)
  - The error message and traceback
  - The Recovery Agent's fix instructions
  - All previous error logs (for context accumulation)
- **Success rate:** High (80%+ of failures are fixable on retry)
- **Example:** `ImportError: no attribute 'APIRoute'` → Fix: "Use `APIRouter` instead of `APIRoute`"

### Tier 2: MODIFY_APPROACH
- **When used:** The fundamental approach is wrong — the algorithm doesn't work, the design pattern is inappropriate
- **What happens:** Recovery Agent suggests a different approach. May also modify test expectations if the test was too tightly coupled to a specific implementation.
- **Success rate:** Medium
- **Example:** "The recursive approach causes stack overflow for large inputs → use iterative approach with explicit stack"

### Tier 3: SKIP_TASK
- **When used:** The task is non-critical, has no downstream dependencies, and all retries are exhausted
- **What happens:** Task is marked as SKIPPED. Pipeline continues to the next task.
- **When NOT to use:** If other tasks depend on this one. In that case, all dependent tasks would also fail.
- **Example:** "Optional pagination feature failed → skip and focus on core CRUD"

### Tier 4: ESCALATE
- **When used:** The error requires human judgment — ambiguous requirements, conflicting constraints, infrastructure issues
- **What happens:** Pipeline pauses. Diagnostic information is presented to the user. User decides: fix manually, provide guidance, skip, or abort.
- **Example:** "The spec says 'use MongoDB' but the tier requires SQL joins. These are contradictory requirements."

## 3.4 Error Context Accumulation

The key innovation in our failure strategy is **progressive context enrichment**. Each failed attempt adds more information for the next attempt:

```
Attempt 1 Context:
├── Specification (what to build)
├── Architecture (how to structure it)
├── Current task (what to implement now)
└── Test files (what behavior is expected)

Attempt 2 Context (everything from attempt 1 PLUS):
├── Error message from attempt 1
├── Full traceback from attempt 1
└── Recovery Agent's diagnosis for attempt 1

Attempt 3 Context (everything from attempts 1-2 PLUS):
├── ALL previous error messages (list)
├── ALL previous tracebacks
├── ALL Recovery Agent diagnoses
└── Specific fix instructions from Recovery Agent

Attempt 4 (if configured):
└── If still failing → SKIP or ESCALATE
```

This pattern works because LLMs are excellent at **learning from examples of what went wrong**. The more error context we provide, the more likely the next attempt produces correct code.

## 3.5 Guardrails — Preventing Dangerous Operations

ForgeAI includes a configurable guardrails system that prevents unsafe operations:

```yaml
guardrails:
  max_files_per_task: 8          # Flag if task generates too many files
  max_lines_per_file: 600        # Warn if file is suspiciously large
  blocked_commands:               # NEVER execute these
    - "rm -rf /"
    - "del /s /q C:\\"
    - "format"
    - "mkfs"
  require_approval_for:
    - "database_schema_changes"   # Schema changes need review
    - "security_sensitive_patterns" # Auth code needs review
    - "external_api_calls"        # External calls need review
```

**Safety constraint (NFR-05):** The FileManager is sandboxed to the project directory. It CANNOT write files outside `./generated_project/`. This is enforced by path validation in `file_manager.py`.

## 3.6 Rollback Support (FR-17)

ForgeAI supports rollback to the last passing checkpoint:
- After each PASSED task, the project state is checkpointed
- If a task fails and the user chooses to abort, the system can roll back to the last known-good state
- This prevents a single bad task from corrupting the entire project

---

# 4. TECH STACK JUSTIFICATION

## 4.1 LLM Choice: Google Gemini 2.5 Flash

### Why Gemini?

**Reason 1: 1 Million Token Context Window**
This is the most important factor. Our agents pass the FULL project state — specification, architecture, all existing files, all test files, error history — to the LLM in a single prompt. For complex Tier 4-5 projects, this context can easily reach 50,000-100,000 tokens. Gemini's 1M token window gives us massive headroom, while GPT-4o (128K) and Claude (200K) would require aggressive truncation.

**Reason 2: Speed**
Gemini 2.5 Flash is optimized for speed. In our pipeline, agents make sequential LLM calls — the total latency is the SUM of all agent calls. Faster inference = faster pipeline completion = better demo experience.

**Reason 3: Free Tier Availability**
Gemini offers a generous free tier, which was essential during our development phase. We could iterate rapidly without worrying about API costs.

**Reason 4: Native JSON Mode**
Gemini supports structured JSON output natively, which reduces parsing failures. Our agents rely on JSON communication — the LLM must return valid JSON that maps to our Pydantic models.

**Reason 5: Google AI Studio Integration**
Easy API key management through Google AI Studio. No complex token setup.

### Comparison Table

| Feature | Gemini 2.5 Flash | GPT-4o | Claude 3.5 Sonnet |
|---------|-------------------|--------|-------------------|
| Context Window | 1,000,000 tokens | 128,000 tokens | 200,000 tokens |
| Speed | ⚡ Sub-second for simple tasks | Fast | Medium |
| Cost | Free tier available | Paid only ($5/1M input) | Paid only ($3/1M input) |
| JSON Mode | Native support | Native support | Prompt-based only |
| Code Quality | Excellent for Python | Excellent | Excellent |
| API Stability | High (Google infra) | High | High |

## 4.2 Why Custom Orchestration (Not LangChain / CrewAI)

### The Decision

We built our orchestration engine from scratch using a custom FSM instead of using LangChain, CrewAI, AutoGen, or similar frameworks.

### Justification

**Reason 1: Full Control Over State Transitions**
Our 16-state FSM with validated transitions gives us precise control over the pipeline flow. With LangChain's AgentExecutor, you lose visibility into state transitions — it's a black box that "figures it out." Our recovery strategy requires explicit state management.

**Reason 2: Failure Recovery Requires Custom Logic**
The 4-tier recovery cascade (Retry → Modify → Skip → Escalate) with error context accumulation is a custom design. LangChain's retry mechanisms are generic — they can't pass accumulated error context, fix instructions, and modified tests back to a specific agent.

**Reason 3: Debuggability**
When something goes wrong (and it will), we need to trace exactly what happened: which state was active, what context was passed, what the LLM returned, why the transition occurred. With LangChain, you're debugging through multiple abstraction layers. With our custom FSM, the code is flat and readable.

**Reason 4: Minimal Dependencies**
Our `requirements.txt` has ~8 direct dependencies. A LangChain-based project would have 50+. Fewer dependencies = faster setup, fewer failure points, easier reproducibility.

**Reason 5: Hackathon Evaluation**
The hackathon explicitly evaluates "Code & Architecture Quality" (10 points). Building the orchestration from scratch demonstrates deep systems engineering — it shows we UNDERSTAND the problem, not just that we can import a library.

## 4.3 Python Ecosystem

### Core Libraries

| Library | Version | Why We Use It |
|---------|---------|---------------|
| `google-generativeai` | Latest | Official Google Gemini Python SDK |
| `pydantic` | v2 | Strict type validation at every data boundary. `AgentContext`, `AgentResult`, `StructuredSpecification`, `AtomicTask`, `ImplementationPlan`, `WorkflowState` — all Pydantic models |
| `click` | 8.x | CLI argument parsing with options, flags, and prompts |
| `rich` | 13.x | Premium terminal UX — progress bars, tables, syntax highlighting, animations |
| `pyyaml` | 6.x | Human-readable configuration (NFR-03: single YAML config file) |
| `python-dotenv` | 1.x | Secure API key loading from `.env` files (no hardcoded keys) |
| `pytest` | 8.x | Industry-standard test framework for generated test suites |
| `fastapi` | 0.110+ | Optional web dashboard with real-time observability |

### Why Pydantic v2 is Critical

Every data contract in ForgeAI is a Pydantic model:
- `AgentContext` — input to every agent
- `AgentResult` — output from every agent
- `StructuredSpecification` — project requirements
- `AtomicTask` — single unit of work
- `ImplementationPlan` — ordered task list
- `WorkflowState` — pipeline state

This means:
- **Type safety** — if an agent returns the wrong type, it fails at the boundary, not deep inside another agent
- **Serialization** — every model can be serialized to JSON/YAML for artifact storage
- **Validation** — fields have constraints (e.g., `tier: int` must be 1-5) that are checked automatically
- **Documentation** — the Pydantic model definitions ARE the documentation for data contracts

## 4.4 Generated Project Constraints

Per hackathon rules:
- **Backend:** Must be Python (FastAPI or Flask)
- **Frontend:** Must be React or Angular (if required by the tier)
- **Testing:** pytest
- **Database:** As appropriate for the tier (SQLite for Tier 1, MongoDB for Tier 5, PostgreSQL for Tier 4)
- **Docker:** Optional but recommended

---

# 5. RISK ASSESSMENT

## 5.1 Our Approach to Risk

We categorize risks on two axes:
- **Impact:** How badly would this risk affect the demo/evaluation if it materialized?
- **Probability:** How likely is it to happen?

We then apply the appropriate strategy:
- **High Impact + High Probability:** MITIGATE NOW — design architecture to prevent it
- **High Impact + Low Probability:** MONITOR — have a contingency plan ready
- **Low Impact + High Probability:** PLAN CONTINGENCY — accept and manage
- **Low Impact + Low Probability:** ACCEPT — don't waste effort

## 5.2 Risk #1: LLM Hallucination (🔴 HIGH Impact, 🔴 HIGH Probability)

**What it is:** The LLM generates code that looks correct but is actually wrong — uses non-existent APIs, invents function parameters, produces logically flawed algorithms.

**Why it's likely:** This is a well-known LLM behavior. Hallucination rate increases with:
- Complex logic (Tier 4-5 projects)
- Less common libraries/patterns
- Long prompts with lots of context

**Our mitigation (multi-layered):**

1. **TDD-First is the primary defense.** Every line of generated code is validated against tests. If the LLM hallucinates an API call, the test will fail, triggering recovery. Without TDD, hallucinated code would silently pass through.

2. **Error context accumulation.** On retry, the LLM sees its own mistake: "You used `fastapi.APIRoute` but that doesn't exist. The correct class is `fastapi.APIRouter`." This dramatically improves the next attempt.

3. **Recovery Agent diagnosis.** Instead of blindly retrying, the Recovery Agent analyzes the error and provides specific fix instructions, grounding the next attempt.

4. **Architecture grounding.** The Architect Agent makes technology decisions upfront. The Coder Agent receives these decisions as context, reducing the chance of using the wrong library.

**Residual risk:** Complex logic errors that pass tests but are algorithmically wrong. Mitigation: the spec-based prompts and security audit catch many of these.

## 5.3 Risk #2: Complex Tier Failure (🔴 HIGH Impact, 🟡 MEDIUM Probability)

**What it is:** Tier 4 (OAuth2/JWT/RBAC) and Tier 5 (MongoDB joins with Change Streams) are significantly harder than Tiers 1-3. The task decomposition might be insufficient, or the generated code might not handle edge cases.

**Why it matters:** The demo tier is revealed at Phase 2 start. If we get Tier 5, we need to handle MongoDB aggregation pipelines, Change Streams, and cross-collection joins — all complex patterns.

**Our mitigation:**

1. **Risk-aware planning.** The Planner Agent assigns HIGH risk to complex tasks. High-risk tasks get more detailed descriptions, more test cases, and potentially more retries.

2. **Graceful degradation.** If a Tier 5 feature fails, the framework doesn't crash — it SKIPS the non-critical task and continues. A mostly-working Tier 5 project scores higher than a crashed pipeline.

3. **Domain-agnostic design.** Our agents don't have tier-specific logic. The Intake Agent determines the tier from the spec, and the architecture/planning agents adjust accordingly. This has been tested across all 5 tiers during development.

4. **Judging criteria favor reliability:** "A team with a reliable Tier 1-2 framework should score higher than a team that attempts Tier 4-5 but produces brittle, frequently failing output."

## 5.4 Risk #3: Test Flakiness (🟡 MEDIUM Impact, 🟡 MEDIUM Probability)

**What it is:** The QA Agent writes tests that are too brittle (testing implementation details instead of behavior) or too loose (passing regardless of code quality).

**Why it matters:** If tests are flaky, the TDD loop breaks down:
- Brittle tests fail even when the production code is correct → wastes retries
- Loose tests pass even when the code is wrong → defeats the purpose of TDD

**Our mitigation:**

1. **QA Agent prompt engineering.** The QA system prompt explicitly instructs: "Write tests that validate BEHAVIOR, not implementation. Use assertions that check outcomes, not internal state."

2. **Recovery Agent can modify tests.** If the Recovery Agent diagnoses that the bug is `error_in: "test_code"`, it can provide modified test code. This allows the system to self-correct bad tests.

3. **Spec grounding.** Both QA and Coder agents receive the structured specification, ensuring tests align with actual requirements.

## 5.5 Risk #4: API Rate Limiting (🟡 MEDIUM Impact, 🟡 MEDIUM Probability)

**What it is:** During intensive code generation (especially for complex tiers with many tasks), the number of LLM API calls can be high. Google's Gemini API has rate limits.

**Why it matters:** Rate limiting during a live demo would cause visible delays or failures.

**Our mitigation:**

1. **LLM Gateway with exponential backoff.** The `LLMGateway` implements automatic retry with increasing delays: 2s → 4s → 8s.

2. **Token tracking.** We track estimated token usage throughout the pipeline. If we're approaching limits, the system can warn.

3. **Efficient prompting.** Agents truncate file contents to 1500 chars per file when building prompts. This keeps token usage manageable while providing enough context.

4. **Free tier generosity.** Gemini's free tier allows 1500 requests per day and 1 million tokens per minute for Flash models — significantly more generous than competitors.

## 5.6 Risk #5: Context Window Overflow (🟡 MEDIUM Impact, 🟢 LOW Probability)

**What it is:** For very large projects, the total context (spec + architecture + existing files + tests + error history) might exceed the LLM's context window.

**Why it's low probability:** Gemini's 1M token context window is enormous. Even a complex Tier 5 project with 20+ files rarely exceeds 50K tokens.

**Our mitigation:**

1. **File truncation.** When existing files are injected into prompts, each file is truncated to 1500 characters. This provides enough context for imports and structure without flooding the context.

2. **Selective context.** Not all files are relevant to every task. Agents receive `existing_files` filtered to the relevant modules.

3. **1M token headroom.** With Gemini 2.5 Flash, we have 10-20x the context window we actually need.

## 5.7 Hardest Parts of Implementation (Honest Assessment)

### Hardest Part 1: Getting TDD Loop Quality Right
The QA Agent must write tests that are **specific enough** to catch bugs but **flexible enough** to accept valid alternative implementations. This balance is hard to achieve with prompt engineering alone. Our Recovery Agent's ability to modify tests is the key safety net.

### Hardest Part 2: Cross-Task Dependency Management
When Task 3 depends on files from Task 1, the Coder Agent for Task 3 needs to read and understand Task 1's output. If Task 1 generated poor-quality code (even if tests pass), Task 3 might struggle. Mitigation: the Architect Agent's upfront design creates consistent patterns that reduce cross-task integration issues.

### Hardest Part 3: Live Demo Reliability
LLMs are inherently non-deterministic. The same prompt can produce different outputs on different runs. A live demo must work reliably. Mitigation:
- Auto-approve mode eliminates human wait times
- Recovery cascade handles most failures automatically
- We'll practice extensively with all 5 tiers before the demo
- Temperature is set to 0.2 (low randomness) for consistency

### Hardest Part 4: MongoDB Joins (Tier 5)
Tier 5 requires implementing inner, left, right, and full outer joins on MongoDB using aggregation pipelines + Change Streams for live data. This is a complex, niche topic that LLMs may hallucinate on. Mitigation: the Intake Agent will produce very detailed spec for Tier 5, and the QA Agent will write thorough test cases that validate join correctness.

---

# APPENDIX A: Key Code Files Reference

| File | Lines | What It Does |
|------|-------|--------------|
| `forgeai/core/orchestrator.py` | 540 | The brain — 16-state FSM, drives entire pipeline |
| `forgeai/agents/base_agent.py` | 129 | Abstract base class — uniform agent contract |
| `forgeai/agents/intake_agent.py` | 170 | NL spec → structured specification |
| `forgeai/agents/architect_agent.py` | ~130 | Designs project structure & architecture |
| `forgeai/agents/planner_agent.py` | ~140 | Creates ordered atomic task list |
| `forgeai/agents/qa_agent.py` | ~130 | TDD-first: writes failing tests |
| `forgeai/agents/coder_agent.py` | ~130 | Generates production code |
| `forgeai/agents/recovery_agent.py` | 107 | Diagnoses failures, recommends recovery |
| `forgeai/agents/security_agent.py` | ~100 | Post-build security audit |
| `forgeai/models/workflow_state.py` | 107 | WorkflowPhase FSM + WorkflowState |
| `forgeai/models/agent_state.py` | 76 | AgentContext + AgentResult contracts |
| `forgeai/models/specification.py` | 83 | StructuredSpecification model |
| `forgeai/models/task.py` | 87 | AtomicTask + ImplementationPlan |
| `forgeai/tools/llm_gateway.py` | 158 | Gemini API with retry & token tracking |
| `forgeai/tools/file_manager.py` | ~130 | Sandboxed file I/O |
| `forgeai/tools/test_runner.py` | ~170 | pytest executor with timeout |
| `forgeai/config/default_config.yaml` | 65 | All guardrails & configuration |
| `forgeai/main.py` | 69 | Entry point — ties CLI + Orchestrator |

---

# APPENDIX B: Requirement Traceability Matrix

| Requirement | Type | Implementation |
|-------------|------|---------------|
| FR-01 | CORE | Intake Agent — accepts NL spec, generates clarifying questions |
| FR-02 | CORE | Intake Agent — produces StructuredSpecification |
| FR-04 | CORE | Architect Agent — designs project structure from scratch |
| FR-05 | CORE | Planner Agent — decomposes into atomic tasks |
| FR-06 | CORE | PLAN_REVIEW checkpoint — user approves plan before execution |
| FR-08 | CORE | Coder Agent — invokes Gemini API for code generation |
| FR-09 | CORE | Diff review callback in Orchestrator |
| FR-11 | CORE | QA Agent — TDD-first test generation |
| FR-12 | CORE | Test Runner — auto-run tests, blocking on failure |
| FR-14 | EXTENDED | Security Agent — AI-powered vulnerability scan |
| FR-15 | CORE | Recovery Agent — auto-retry with error context |
| FR-17 | EXTENDED | Checkpoint rollback in WorkflowState |
| NFR-01 | Usability | CLI with Rich — intuitive setup and usage |
| NFR-02 | Transparency | ActivityLogger — append-only log of every action |
| NFR-03 | Configurability | ConfigManager + default_config.yaml |
| NFR-04 | Portability | Pure Python, no cloud deps beyond Gemini API |
| NFR-05 | Safety | FileManager sandboxing + blocked commands |
| NFR-06 | Observability | WorkflowState.get_summary() + workflow_summary.json |

---

*Document last updated: April 10, 2026*
*ForgeAI — Itlanta Hackathon 2026 | Phase 1 Design Submission*
