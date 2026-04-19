"""
ForgeAI — Architecture Diagram Generator
Generates all key diagrams using Python Graphviz.
Run: pip install graphviz   then   python generate_diagrams.py
Output: PDF/PNG files in the current directory.
"""

import graphviz


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: 16-State Finite State Machine
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fsm_diagram():
    """Generate the 16-state FSM workflow diagram."""
    dot = graphviz.Digraph(
        name="ForgeAI_FSM",
        comment="ForgeAI 16-State Finite State Machine",
        format="png",
        engine="dot",
    )

    # Graph attributes
    dot.attr(
        rankdir="TB",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.6",
        ranksep="0.8",
        label="ForgeAI — 16-State Finite State Machine\nWorkflow Orchestration Pipeline",
        labelloc="t",
        labeljust="c",
        style="filled",
        fillcolor="#0a0e1a",
    )

    # Default node style
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="11",
        fontcolor="#f1f5f9",
        penwidth="1.5",
        margin="0.15,0.1",
    )

    # Default edge style
    dot.attr(
        "edge",
        color="#64748b",
        fontname="Helvetica",
        fontsize="9",
        fontcolor="#94a3b8",
        arrowsize="0.7",
        penwidth="1.2",
    )

    # ── Start node ──
    dot.node("start", "", shape="doublecircle", width="0.3", height="0.3",
             fillcolor="#64748b", color="#64748b")

    # ── Requirements Phase (Blue) ──
    blue_fill = "#1e3a5f"
    blue_border = "#3b82f6"
    dot.node("IDLE", "IDLE", fillcolor=blue_fill, color=blue_border)
    dot.node("INTAKE", "INTAKE\n(Intake Agent)", fillcolor=blue_fill, color=blue_border)
    dot.node("CLARIFICATION", "CLARIFICATION\n(Ask User Questions)", fillcolor="#3d2e0a", color="#f59e0b")
    dot.node("SPECIFICATION", "SPECIFICATION\n(Structured Spec)", fillcolor="#0a3d2e", color="#10b981")

    # ── Design Phase (Purple) ──
    purple_fill = "#2d1b5e"
    purple_border = "#8b5cf6"
    dot.node("ARCHITECTURE", "ARCHITECTURE\n(Architect Agent)", fillcolor=purple_fill, color=purple_border)
    dot.node("PLANNING", "PLANNING\n(Planner Agent)", fillcolor=purple_fill, color=purple_border)

    # ── Checkpoint (Orange dashed) ──
    dot.node("PLAN_REVIEW", "🔒 PLAN_REVIEW\n(Human Checkpoint)", fillcolor="#3d2e0a",
             color="#f59e0b", style="filled,rounded,dashed", penwidth="2.5")

    # ── TDD Execution Loop (Green/Cyan) ──
    green_fill = "#0a3d2e"
    green_border = "#10b981"
    cyan_fill = "#0a2d3d"
    cyan_border = "#06b6d4"
    orange_fill = "#3d2e0a"
    orange_border = "#f59e0b"

    dot.node("EXECUTION", "EXECUTION\n(Next Task from Plan)", fillcolor=green_fill, color=green_border)
    dot.node("TASK_QA", "TASK_QA\n✅ QA Writes Tests", fillcolor=orange_fill, color=orange_border)
    dot.node("TASK_CODE", "TASK_CODE\n💻 Coder Generates Code", fillcolor=blue_fill, color=blue_border)
    dot.node("TASK_TEST", "TASK_TEST\n🧪 Run pytest", fillcolor=cyan_fill, color=cyan_border)

    # ── Recovery (Red) ──
    red_fill = "#3d1a1a"
    red_border = "#ef4444"
    dot.node("TASK_RECOVERY", "TASK_RECOVERY\n🔄 Recovery Agent", fillcolor=red_fill, color=red_border)

    # ── Post-execution (Purple) ──
    dot.node("SECURITY_AUDIT", "SECURITY_AUDIT\n🔒 Security Agent", fillcolor=purple_fill, color=purple_border)
    dot.node("SUMMARY", "SUMMARY\n📊 Generate Report", fillcolor=blue_fill, color=blue_border)
    dot.node("DONE", "✅ DONE", fillcolor=green_fill, color=green_border,
             penwidth="2.5", fontsize="13")
    dot.node("ERROR", "❌ ERROR", fillcolor=red_fill, color=red_border)

    # ── Edges: Requirements Phase ──
    dot.edge("start", "IDLE", color="#64748b")
    dot.edge("IDLE", "INTAKE", label="user submits spec")
    dot.edge("INTAKE", "CLARIFICATION", label="ambiguities\nfound", color="#f59e0b")
    dot.edge("INTAKE", "SPECIFICATION", label="spec is clear", color="#10b981")
    dot.edge("CLARIFICATION", "SPECIFICATION", label="answers\nreceived", color="#10b981")
    dot.edge("CLARIFICATION", "INTAKE", label="more questions", style="dashed", color="#64748b")

    # ── Edges: Design Phase ──
    dot.edge("SPECIFICATION", "ARCHITECTURE", label="spec finalized")
    dot.edge("ARCHITECTURE", "PLANNING", label="design approved")
    dot.edge("PLANNING", "PLAN_REVIEW", label="plan generated")
    dot.edge("PLAN_REVIEW", "EXECUTION", label="✅ approved", color="#10b981", fontcolor="#10b981", penwidth="2")
    dot.edge("PLAN_REVIEW", "PLANNING", label="❌ revisions", color="#ef4444", fontcolor="#ef4444", style="dashed")

    # ── Edges: TDD Execution Loop ──
    # Cluster for TDD loop
    with dot.subgraph(name="cluster_tdd") as tdd:
        tdd.attr(
            label="TDD Execution Loop (per task)",
            style="dashed,rounded",
            color="#ec4899",
            fontcolor="#ec4899",
            fontname="Helvetica",
            fontsize="11",
            bgcolor="#0d0a1a",
            penwidth="1.5",
        )

    dot.edge("EXECUTION", "TASK_QA", label="next task", color="#10b981")
    dot.edge("TASK_QA", "TASK_CODE", label="tests written")
    dot.edge("TASK_CODE", "TASK_TEST", label="code generated")
    dot.edge("TASK_TEST", "EXECUTION", label="✅ tests pass", color="#10b981", fontcolor="#10b981", penwidth="2")
    dot.edge("TASK_TEST", "TASK_RECOVERY", label="❌ tests fail", color="#ef4444", fontcolor="#ef4444", penwidth="2")
    dot.edge("TASK_RECOVERY", "TASK_CODE", label="retry with fix", color="#f59e0b", fontcolor="#f59e0b", style="dashed")
    dot.edge("TASK_RECOVERY", "EXECUTION", label="skip task", color="#64748b", style="dashed")
    dot.edge("TASK_RECOVERY", "ERROR", label="escalate", color="#ef4444", style="dashed")

    # ── Edges: Post-execution ──
    dot.edge("EXECUTION", "SECURITY_AUDIT", label="all tasks done", color="#8b5cf6")
    dot.edge("EXECUTION", "SUMMARY", label="(skip audit)", style="dashed", color="#64748b")
    dot.edge("SECURITY_AUDIT", "SUMMARY", label="audit complete")
    dot.edge("SUMMARY", "DONE", label="report generated", color="#10b981", penwidth="2")
    dot.edge("ERROR", "IDLE", label="reset", style="dashed", color="#ef4444")

    dot.render("diagram_fsm", cleanup=True)
    print("✅ Generated: diagram_fsm.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: 4-Tier Failure Recovery Cascade
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recovery_diagram():
    """Generate the 4-tier cascading recovery flowchart."""
    dot = graphviz.Digraph(
        name="ForgeAI_Recovery",
        comment="ForgeAI 4-Tier Recovery Cascade",
        format="png",
        engine="dot",
    )

    dot.attr(
        rankdir="TB",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.5",
        ranksep="0.7",
        label="ForgeAI — 4-Tier Cascading Recovery System\nFailure Detection → Diagnosis → Strategy → Resolution",
        labelloc="t",
        labeljust="c",
    )

    dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
             fontsize="11", fontcolor="#f1f5f9", penwidth="1.5", margin="0.2,0.12")
    dot.attr("edge", color="#64748b", fontname="Helvetica", fontsize="9",
             fontcolor="#94a3b8", arrowsize="0.7", penwidth="1.2")

    # ── Trigger ──
    dot.node("FAIL", "❌ Test Failure Detected",
             fillcolor="#5c1a1a", color="#ef4444", penwidth="2.5", fontsize="13")

    # ── Diagnosis ──
    dot.node("DIAG", "🔍 Recovery Agent\nDiagnoses Root Cause",
             fillcolor="#1e3a5f", color="#3b82f6", fontsize="12")

    # ── Classification ──
    dot.node("CLASSIFY", "Error Classification\n(root_cause, error_type, error_in)",
             shape="diamond", fillcolor="#2d1b5e", color="#8b5cf6",
             fontsize="10", width="3", height="1.2")

    # ── 4 Strategies ──
    dot.node(
        "FIX",
        "🔧 RETRY_WITH_FIX\n━━━━━━━━━━━━━━━━━━\n• Specific fix instructions\n• Error + traceback → Coder\n• Modify test if bug in test\n• Confidence: HIGH",
        fillcolor="#0a3d2e", color="#10b981", fontsize="10",
    )
    dot.node(
        "MODIFY",
        "🔄 MODIFY_APPROACH\n━━━━━━━━━━━━━━━━━━\n• Different algorithm/pattern\n• Revised test expectations\n• Fresh code generation\n• Confidence: MEDIUM",
        fillcolor="#3d2e0a", color="#f59e0b", fontsize="10",
    )
    dot.node(
        "SKIP",
        "⏭️ SKIP_TASK\n━━━━━━━━━━━━━━━━━━\n• Task is non-blocking\n• No downstream deps\n• Log and proceed\n• Confidence: N/A",
        fillcolor="#1a2433", color="#64748b", fontsize="10",
    )
    dot.node(
        "ESCALATE",
        "🚨 ESCALATE\n━━━━━━━━━━━━━━━━━━\n• Pause pipeline\n• Diagnostics → User\n• User decides action\n• Confidence: LOW",
        fillcolor="#5c1a1a", color="#ef4444", fontsize="10",
    )

    # ── Decision & Loop nodes ──
    dot.node("RETRY_CHECK", "Attempt ≤\nMax Retries?",
             shape="diamond", fillcolor="#2d1b5e", color="#8b5cf6",
             fontsize="10", width="1.5", height="0.8")
    dot.node("CODER", "💻 Coder Agent\n(with fix context + error history)",
             fillcolor="#1e3a5f", color="#3b82f6")
    dot.node("TEST", "🧪 Run Tests",
             fillcolor="#0a2d3d", color="#06b6d4")
    dot.node("DONE", "✅ Task Complete",
             fillcolor="#0a3d2e", color="#10b981", penwidth="2.5", fontsize="13")

    # ── Edges ──
    dot.edge("FAIL", "DIAG")
    dot.edge("DIAG", "CLASSIFY")

    dot.edge("CLASSIFY", "FIX", label="syntax / import\nerrors", color="#10b981", fontcolor="#10b981")
    dot.edge("CLASSIFY", "MODIFY", label="logic / approach\nerrors", color="#f59e0b", fontcolor="#f59e0b")
    dot.edge("CLASSIFY", "SKIP", label="non-critical\ntask", color="#64748b")
    dot.edge("CLASSIFY", "ESCALATE", label="requires human\njudgment", color="#ef4444", fontcolor="#ef4444")

    dot.edge("FIX", "RETRY_CHECK")
    dot.edge("MODIFY", "RETRY_CHECK")

    dot.edge("RETRY_CHECK", "CODER", label="Yes", color="#10b981", fontcolor="#10b981", penwidth="2")
    dot.edge("RETRY_CHECK", "SKIP", label="No (exhausted)", color="#ef4444", fontcolor="#ef4444", style="dashed")

    dot.edge("CODER", "TEST")
    dot.edge("TEST", "DONE", label="Pass ✅", color="#10b981", fontcolor="#10b981", penwidth="2")
    dot.edge("TEST", "DIAG", label="Fail ❌", color="#ef4444", fontcolor="#ef4444",
             style="dashed", constraint="false")

    dot.render("diagram_recovery", cleanup=True)
    print("✅ Generated: diagram_recovery.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: High-Level System Architecture (4 Layers)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_architecture_diagram():
    """Generate the 4-layer system architecture diagram."""
    dot = graphviz.Digraph(
        name="ForgeAI_Architecture",
        comment="ForgeAI System Architecture",
        format="png",
        engine="dot",
    )

    dot.attr(
        rankdir="TB",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.4",
        ranksep="0.6",
        label="ForgeAI — 4-Layer System Architecture",
        labelloc="t",
        compound="true",
    )

    dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
             fontsize="10", fontcolor="#f1f5f9", penwidth="1.5", margin="0.15,0.08")
    dot.attr("edge", color="#475569", fontname="Helvetica", fontsize="8",
             arrowsize="0.6", penwidth="1")

    # ── Layer 1: UI ──
    with dot.subgraph(name="cluster_ui") as ui:
        ui.attr(label="🖥️  UI LAYER", style="filled,rounded,dashed",
                fillcolor="#111d33", color="#3b82f6", fontcolor="#3b82f6",
                fontname="Helvetica", fontsize="12", penwidth="1.5")
        ui.node("CLI", "🖥️ Rich CLI\nInteractive Terminal", fillcolor="#1e3a5f", color="#3b82f6")
        ui.node("WEB", "🌐 FastAPI Dashboard\nReal-time Observability", fillcolor="#1e3a5f", color="#3b82f6")

    # ── Layer 2: Orchestration ──
    with dot.subgraph(name="cluster_orch") as orch:
        orch.attr(label="🧠  ORCHESTRATION LAYER", style="filled,rounded,dashed",
                  fillcolor="#1a1133", color="#8b5cf6", fontcolor="#8b5cf6",
                  fontname="Helvetica", fontsize="12", penwidth="1.5")
        orch.node("ORC", "🧠 Orchestrator\n16-State FSM Engine", fillcolor="#2d1b5e", color="#8b5cf6",
                  fontsize="12", penwidth="2")
        orch.node("CFG", "⚙️ Config Manager\nYAML Guardrails", fillcolor="#2d1b5e", color="#8b5cf6")
        orch.node("LOG", "📝 Activity Logger\nAppend-Only Log", fillcolor="#2d1b5e", color="#8b5cf6")
        orch.node("WFS", "📦 Workflow State\nPydantic Models", fillcolor="#2d1b5e", color="#8b5cf6")

    # ── Layer 3: Agents ──
    with dot.subgraph(name="cluster_agents") as agents:
        agents.attr(label="🤖  AGENT LAYER — 7 Specialized Agents", style="filled,rounded,dashed",
                    fillcolor="#0d2218", color="#10b981", fontcolor="#10b981",
                    fontname="Helvetica", fontsize="12", penwidth="1.5")
        agents.node("A_INT", "🎯 Intake\nAgent", fillcolor="#1e3a5f", color="#3b82f6")
        agents.node("A_ARC", "🏗️ Architect\nAgent", fillcolor="#0a3d2e", color="#10b981")
        agents.node("A_PLN", "📋 Planner\nAgent", fillcolor="#0a3d2e", color="#10b981")
        agents.node("A_QA", "✅ QA Agent\n(TDD-First)", fillcolor="#3d2e0a", color="#f59e0b")
        agents.node("A_COD", "💻 Coder\nAgent", fillcolor="#3d2e0a", color="#f59e0b")
        agents.node("A_SEC", "🔒 Security\nAgent", fillcolor="#2d1b5e", color="#8b5cf6")
        agents.node("A_REC", "🔄 Recovery\nAgent", fillcolor="#3d1a1a", color="#ef4444")

    # ── Layer 4: Tools ──
    with dot.subgraph(name="cluster_tools") as tools:
        tools.attr(label="🔧  TOOL LAYER", style="filled,rounded,dashed",
                   fillcolor="#0d1f2d", color="#06b6d4", fontcolor="#06b6d4",
                   fontname="Helvetica", fontsize="12", penwidth="1.5")
        tools.node("T_LLM", "🤖 LLM Gateway\nGemini API + Retry", fillcolor="#0a2d3d", color="#06b6d4")
        tools.node("T_FM", "📂 File Manager\nSandboxed I/O", fillcolor="#0a2d3d", color="#06b6d4")
        tools.node("T_TR", "🧪 Test Runner\npytest", fillcolor="#0a2d3d", color="#06b6d4")
        tools.node("T_DK", "🐳 Docker Builder\nCompose", fillcolor="#0a2d3d", color="#06b6d4")

    # ── Edges: UI → Orchestrator ──
    dot.edge("CLI", "ORC", lhead="cluster_orch")
    dot.edge("WEB", "ORC", lhead="cluster_orch")

    # ── Edges: Orchestrator → Agents ──
    dot.edge("ORC", "A_INT", label="invoke", color="#8b5cf6")
    dot.edge("ORC", "A_ARC", color="#8b5cf6")
    dot.edge("ORC", "A_PLN", color="#8b5cf6")
    dot.edge("ORC", "A_QA", color="#8b5cf6")
    dot.edge("ORC", "A_COD", color="#8b5cf6")
    dot.edge("ORC", "A_SEC", color="#8b5cf6")
    dot.edge("ORC", "A_REC", color="#8b5cf6")

    # ── Edges: Agents → Tools ──
    dot.edge("A_INT", "T_LLM", color="#06b6d4")
    dot.edge("A_ARC", "T_LLM", color="#06b6d4")
    dot.edge("A_PLN", "T_LLM", color="#06b6d4")
    dot.edge("A_QA", "T_LLM", color="#06b6d4")
    dot.edge("A_COD", "T_LLM", color="#06b6d4")
    dot.edge("A_SEC", "T_LLM", color="#06b6d4")
    dot.edge("A_REC", "T_LLM", color="#06b6d4")

    # ── Edges: Orchestrator → Tools ──
    dot.edge("ORC", "T_FM", label="write files", color="#06b6d4", style="dashed")
    dot.edge("ORC", "T_TR", label="run tests", color="#06b6d4", style="dashed")
    dot.edge("ORC", "T_DK", label="build docker", color="#06b6d4", style="dashed")

    # ── Orchestrator internal ──
    dot.edge("ORC", "CFG", style="dotted", color="#64748b", arrowhead="none")
    dot.edge("ORC", "LOG", style="dotted", color="#64748b", arrowhead="none")
    dot.edge("ORC", "WFS", style="dotted", color="#64748b", arrowhead="none")

    dot.render("diagram_architecture", cleanup=True)
    print("✅ Generated: diagram_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4: Agent Data Flow (AgentContext → BaseAgent → AgentResult)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_agent_flow_diagram():
    """Generate the agent data flow / contract diagram."""
    dot = graphviz.Digraph(
        name="ForgeAI_AgentFlow",
        comment="ForgeAI Agent Data Flow",
        format="png",
        engine="dot",
    )

    dot.attr(
        rankdir="LR",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.6",
        ranksep="1.2",
        label="ForgeAI — Agent Uniform Contract & Data Flow\nEvery Agent: AgentContext → BaseAgent → AgentResult",
        labelloc="t",
    )

    dot.attr("node", shape="record", style="filled,rounded", fontname="Helvetica",
             fontsize="10", fontcolor="#f1f5f9", penwidth="1.5")
    dot.attr("edge", color="#64748b", fontname="Helvetica", fontsize="9",
             arrowsize="0.7", penwidth="1.5")

    # ── AgentContext (Input) ──
    dot.node(
        "CTX",
        "{ 📥 AgentContext (Input) |"
        " role: AgentRole |"
        " specification: StructuredSpec |"
        " architecture: dict |"
        " current_task: AtomicTask |"
        " existing_files: \\{path → content\\} |"
        " error_message: str |"
        " error_traceback: str |"
        " previous_attempts: list[str] |"
        " user_input: str |"
        " clarification_responses: dict |"
        " retry_count: int }",
        fillcolor="#1e3a5f",
        color="#3b82f6",
    )

    # ── BaseAgent (Processor) ──
    dot.node(
        "AGENT",
        "{ 🤖 BaseAgent |"
        " build_system_prompt() |"
        " build_user_prompt(ctx) |"
        " → LLM Gateway → |"
        " parse_response(raw) }",
        fillcolor="#2d1b5e",
        color="#8b5cf6",
        fontsize="11",
    )

    # ── AgentResult (Output) ──
    dot.node(
        "RES",
        "{ 📤 AgentResult (Output) |"
        " success: bool |"
        " specification: StructuredSpec |"
        " architecture: dict |"
        " implementation_plan: Plan |"
        " generated_files: \\{path → code\\} |"
        " test_results: dict |"
        " security_report: dict |"
        " clarifying_questions: list |"
        " requires_human_input: bool |"
        " api_calls_made: int |"
        " error: str }",
        fillcolor="#0a3d2e",
        color="#10b981",
    )

    # ── Edges ──
    dot.edge("CTX", "AGENT", label="  context  ", color="#3b82f6", fontcolor="#3b82f6", penwidth="2")
    dot.edge("AGENT", "RES", label="  result  ", color="#10b981", fontcolor="#10b981", penwidth="2")

    dot.render("diagram_agent_flow", cleanup=True)
    print("✅ Generated: diagram_agent_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5: TDD Execution Pipeline (per task)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tdd_pipeline_diagram():
    """Generate the TDD execution pipeline diagram."""
    dot = graphviz.Digraph(
        name="ForgeAI_TDD",
        comment="ForgeAI TDD-First Execution Pipeline",
        format="png",
        engine="dot",
    )

    dot.attr(
        rankdir="LR",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.6",
        ranksep="0.8",
        label="ForgeAI — TDD-First Execution Pipeline (Per Task)\nQA → Coder → Test → Recovery Loop",
        labelloc="t",
    )

    dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
             fontsize="11", fontcolor="#f1f5f9", penwidth="1.5", margin="0.2,0.15")
    dot.attr("edge", color="#64748b", fontname="Helvetica", fontsize="9",
             fontcolor="#94a3b8", arrowsize="0.7", penwidth="1.2")

    # ── Nodes ──
    dot.node("ORCH", "🧠 Orchestrator\nget_next_task()\ncheck dependencies",
             fillcolor="#2d1b5e", color="#8b5cf6", fontsize="10")

    dot.node("QA", "✅ STEP 1: QA Agent\n━━━━━━━━━━━━━━\nWrite FAILING tests\nBEFORE any code\n→ tests/test_*.py",
             fillcolor="#0a3d2e", color="#10b981", fontsize="10")

    dot.node("CODER", "💻 STEP 2: Coder Agent\n━━━━━━━━━━━━━━━━━\nGenerate production code\nto PASS failing tests\n→ src/*.py",
             fillcolor="#1e3a5f", color="#3b82f6", fontsize="10")

    dot.node("TEST", "🧪 STEP 3: Test Runner\n━━━━━━━━━━━━━━━━━\nExecute pytest suite\nCollect pass/fail results",
             fillcolor="#0a2d3d", color="#06b6d4", fontsize="10")

    dot.node("PASS", "✅ TASK PASSED\nMove to next task",
             fillcolor="#0a3d2e", color="#10b981", penwidth="2.5", fontsize="12")

    dot.node("RECOVERY", "🔄 STEP 4: Recovery Agent\n━━━━━━━━━━━━━━━━━━━\nDiagnose root cause\nClassify error type\nChoose strategy:\nRETRY | MODIFY | SKIP | ESCALATE",
             fillcolor="#3d1a1a", color="#ef4444", fontsize="10")

    dot.node("CONTEXT", "📊 Error Context\nAccumulation\n━━━━━━━━━━━━━━━━\nAttempt 1: spec+arch+task\nAttempt 2: +error+traceback\nAttempt 3: +fix instructions\nAttempt 4: SKIP/ESCALATE",
             fillcolor="#1a1a2e", color="#64748b", fontsize="9", shape="note")

    # ── Edges ──
    dot.edge("ORCH", "QA", label="task + spec\n+ architecture", color="#10b981", penwidth="2")
    dot.edge("QA", "CODER", label="tests written\nto disk", color="#10b981")
    dot.edge("CODER", "TEST", label="code written\nto disk")
    dot.edge("TEST", "PASS", label="ALL PASS ✅", color="#10b981", fontcolor="#10b981", penwidth="2.5")
    dot.edge("TEST", "RECOVERY", label="FAIL ❌", color="#ef4444", fontcolor="#ef4444", penwidth="2")
    dot.edge("RECOVERY", "CODER", label="retry with\nfix instructions", color="#f59e0b",
             fontcolor="#f59e0b", style="dashed", penwidth="1.5")
    dot.edge("RECOVERY", "CONTEXT", style="dotted", color="#64748b", arrowhead="none")

    dot.render("diagram_tdd_pipeline", cleanup=True)
    print("✅ Generated: diagram_tdd_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 6: Agent Pipeline (Sequential Flow)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_agent_pipeline_diagram():
    """Generate the agent sequential pipeline diagram."""
    dot = graphviz.Digraph(
        name="ForgeAI_Pipeline",
        comment="ForgeAI Agent Pipeline",
        format="png",
        engine="dot",
    )

    dot.attr(
        rankdir="LR",
        bgcolor="#0a0e1a",
        fontcolor="#f1f5f9",
        fontname="Helvetica",
        fontsize="14",
        pad="0.5",
        nodesep="0.5",
        ranksep="0.7",
        label="ForgeAI — Agent Pipeline & Artifact Flow\nEach agent produces artifacts consumed by downstream agents",
        labelloc="t",
    )

    dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
             fontsize="10", fontcolor="#f1f5f9", penwidth="1.5", margin="0.15,0.1")
    dot.attr("edge", color="#64748b", fontname="Helvetica", fontsize="8",
             fontcolor="#94a3b8", arrowsize="0.7", penwidth="1.5")

    # ── User ──
    dot.node("USER", "🧑 User\nNatural Language\nSpecification",
             fillcolor="#1e3a5f", color="#3b82f6", shape="ellipse", fontsize="11")

    # ── Agents ──
    dot.node("INT", "🎯 Intake Agent\n→ StructuredSpec", fillcolor="#1e3a5f", color="#3b82f6")
    dot.node("ARC", "🏗️ Architect Agent\n→ Architecture dict", fillcolor="#0a3d2e", color="#10b981")
    dot.node("PLN", "📋 Planner Agent\n→ ImplementationPlan", fillcolor="#0a3d2e", color="#10b981")
    dot.node("CP", "🔒 CHECKPOINT\nUser Approval", fillcolor="#3d2e0a", color="#f59e0b",
             style="filled,rounded,dashed", penwidth="2.5")
    dot.node("QA", "✅ QA Agent\n→ Test Files", fillcolor="#3d2e0a", color="#f59e0b")
    dot.node("COD", "💻 Coder Agent\n→ Production Code", fillcolor="#3d2e0a", color="#f59e0b")
    dot.node("SEC", "🔒 Security Agent\n→ Audit Report", fillcolor="#2d1b5e", color="#8b5cf6")
    dot.node("OUT", "📊 Summary Report\n+ Generated Project",
             fillcolor="#0a3d2e", color="#10b981", shape="ellipse", fontsize="11", penwidth="2.5")

    # ── Edges ──
    dot.edge("USER", "INT", label="raw spec", color="#3b82f6", penwidth="2")
    dot.edge("INT", "ARC", label="StructuredSpec", color="#10b981")
    dot.edge("ARC", "PLN", label="architecture", color="#10b981")
    dot.edge("PLN", "CP", label="plan", color="#f59e0b")
    dot.edge("CP", "QA", label="approved ✅", color="#10b981", penwidth="2")
    dot.edge("QA", "COD", label="test files", color="#f59e0b")
    dot.edge("COD", "SEC", label="all code", color="#8b5cf6")
    dot.edge("SEC", "OUT", label="reports", color="#10b981", penwidth="2")

    # ── Recovery loop ──
    dot.node("REC", "🔄 Recovery", fillcolor="#3d1a1a", color="#ef4444", fontsize="9")
    dot.edge("COD", "REC", label="test fail", color="#ef4444", style="dashed")
    dot.edge("REC", "COD", label="retry", color="#f59e0b", style="dashed")

    dot.render("diagram_agent_pipeline", cleanup=True)
    print("✅ Generated: diagram_agent_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Generate all diagrams
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI — Generating Architecture Diagrams")
    print("=" * 60)

    generate_fsm_diagram()
    generate_recovery_diagram()
    generate_architecture_diagram()
    generate_agent_flow_diagram()
    generate_tdd_pipeline_diagram()
    generate_agent_pipeline_diagram()

    print("\n" + "=" * 60)
    print("✅ All 6 diagrams generated successfully!")
    print("Files: diagram_fsm.png, diagram_recovery.png,")
    print("       diagram_architecture.png, diagram_agent_flow.png,")
    print("       diagram_tdd_pipeline.png, diagram_agent_pipeline.png")
    print("=" * 60)
