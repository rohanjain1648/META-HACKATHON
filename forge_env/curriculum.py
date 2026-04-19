"""ForgeRL Adaptive Curriculum — Self-improvement through escalating difficulty.

Implements the Self-Improvement hackathon theme by providing:
1. A bank of project specifications across 5 tiers of difficulty
2. Adaptive tier selection based on agent performance history
3. Randomized spec variations to prevent memorization
4. Performance tracking with promotion/demotion thresholds

Tier progression:
  Tier 1: Basic CRUD (The Ledger) — ~50 steps
  Tier 2: Business Rules (Logic Engine) — ~80 steps
  Tier 3: API Integration (Live Bridge) — ~120 steps  
  Tier 4: Auth/RBAC (The Gatekeeper) — ~180 steps
  Tier 5: Complex DB (Mongo-SQL Engine) — ~300 steps
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SpecVariation(BaseModel):
    """A single project specification with randomizable parameters."""

    name: str
    tier: int
    base_description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    tech_variations: dict[str, list[str]] = Field(default_factory=dict)
    constraint_pool: list[str] = Field(default_factory=list)
    max_steps: int = 100
    expected_tasks: int = 8


class EpisodeResult(BaseModel):
    """Result of a single training episode."""

    spec_name: str
    tier: int
    success: bool
    total_reward: float
    steps_taken: int
    tasks_completed: int
    tasks_total: int
    test_pass_rate: float


# ── Spec Bank ─────────────────────────────────────────────────────────────────

SPEC_BANK: list[SpecVariation] = [
    # ── Tier 1: Basic CRUD ──
    SpecVariation(
        name="expense_ledger",
        tier=1,
        base_description=(
            "Build a REST API for personal expense tracking. Users can create, "
            "read, update, and delete expense records. Each expense has a date, "
            "amount, category, and description. The API must validate all inputs "
            "and return proper HTTP status codes."
        ),
        acceptance_criteria=[
            "POST /expenses creates a new expense with validation",
            "GET /expenses returns a paginated list of expenses",
            "GET /expenses/{id} returns a single expense",
            "PUT /expenses/{id} updates an existing expense",
            "DELETE /expenses/{id} removes an expense",
            "All endpoints return proper HTTP status codes",
            "Invalid inputs return 400/422 with error details",
        ],
        tech_variations={
            "database": ["sqlite", "json_file"],
            "framework": ["fastapi"],
        },
        constraint_pool=[
            "Must support filtering by category",
            "Must support date range queries",
            "Must include summary endpoint for totals by category",
        ],
        max_steps=80,
        expected_tasks=8,
    ),
    SpecVariation(
        name="todo_api",
        tier=1,
        base_description=(
            "Build a task management REST API. Users can manage todo items "
            "with title, description, priority (low/medium/high), due date, "
            "and completion status. Support batch operations."
        ),
        acceptance_criteria=[
            "CRUD endpoints for todo items",
            "Priority-based sorting",
            "Due date validation (not in the past)",
            "Batch mark-as-complete endpoint",
            "Filter by status and priority",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support tags/labels on todos",
            "Must include a statistical summary endpoint",
        ],
        max_steps=80,
        expected_tasks=7,
    ),
    SpecVariation(
        name="inventory_tracker",
        tier=1,
        base_description=(
            "Build an inventory management API for a small warehouse. "
            "Track products with name, SKU, quantity, price, and reorder level. "
            "Alert when stock falls below reorder level."
        ),
        acceptance_criteria=[
            "CRUD for products",
            "Stock adjustment endpoint (add/remove quantity)",
            "Low stock alert endpoint",
            "Search by name or SKU",
            "Input validation for all fields",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must track stock adjustment history",
            "Must support bulk import via CSV",
        ],
        max_steps=80,
        expected_tasks=8,
    ),
    # ── Tier 2: Business Logic ──
    SpecVariation(
        name="tax_calculator",
        tier=2,
        base_description=(
            "Build a regional tax/pricing calculator API. Support multiple tax "
            "jurisdictions with different rates, exemptions, and compound rules. "
            "Calculate total tax for a shopping cart with items from different "
            "categories. Handle edge cases like tax-exempt items and bulk discounts."
        ),
        acceptance_criteria=[
            "POST /calculate accepts a cart and jurisdiction, returns itemized tax",
            "Support at least 3 tax jurisdictions with different rules",
            "Handle tax-exempt categories",
            "Support compound tax (tax on tax) for specific jurisdictions",
            "Bulk discount logic (buy 3 get 10% off)",
            "Proper rounding to 2 decimal places",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support date-effective tax rates",
            "Must handle currency conversion",
            "Must generate tax receipt PDF",
        ],
        max_steps=120,
        expected_tasks=12,
    ),
    SpecVariation(
        name="pricing_engine",
        tier=2,
        base_description=(
            "Build a dynamic pricing engine for an e-commerce platform. "
            "Prices change based on demand, inventory levels, competitor pricing, "
            "time of day, and customer loyalty tier. Implement a rules engine "
            "that evaluates pricing rules in priority order."
        ),
        acceptance_criteria=[
            "Define pricing rules via API",
            "Rules have priority ordering",
            "Support conditions: inventory < N, time between X-Y, loyalty_tier = Z",
            "POST /price evaluates all rules and returns final price",
            "Audit trail for price decisions",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support A/B testing of pricing strategies",
            "Must enforce minimum margin constraints",
        ],
        max_steps=120,
        expected_tasks=11,
    ),
    # ── Tier 3: API Integration ──
    SpecVariation(
        name="weather_dashboard_api",
        tier=3,
        base_description=(
            "Build a weather aggregation API that fetches data from multiple "
            "weather providers (simulated), normalizes the data, caches results, "
            "and provides a unified query interface. Handle API rate limits, "
            "timeouts, and provider failover gracefully."
        ),
        acceptance_criteria=[
            "Fetch weather from at least 2 simulated providers",
            "Normalize different response formats to a common schema",
            "Cache results with configurable TTL",
            "Failover to backup provider on primary failure",
            "Rate limiting on outbound requests",
            "Async request handling",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support webhook notifications for severe weather",
            "Must include a health check for each provider",
        ],
        max_steps=150,
        expected_tasks=14,
    ),
    SpecVariation(
        name="payment_gateway",
        tier=3,
        base_description=(
            "Build a payment processing gateway that integrates with simulated "
            "payment providers (Stripe-like, PayPal-like). Handle payment "
            "intents, webhooks, idempotency, and refunds. Implement retry "
            "logic for failed transactions."
        ),
        acceptance_criteria=[
            "Create payment intent endpoint",
            "Process payment with simulated provider",
            "Handle webhooks for payment status updates",
            "Idempotent payment processing",
            "Refund endpoint with partial refund support",
            "Transaction history with filtering",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support multiple currencies",
            "Must implement fraud detection heuristics",
        ],
        max_steps=150,
        expected_tasks=15,
    ),
    # ── Tier 4: Auth & RBAC ──
    SpecVariation(
        name="auth_service",
        tier=4,
        base_description=(
            "Build a full OAuth2/JWT authentication service with Role-Based "
            "Access Control (RBAC). Support user registration, login, token "
            "refresh, role assignment, and permission-based route protection. "
            "Implement password hashing, token blacklisting, and session management."
        ),
        acceptance_criteria=[
            "User registration with email validation",
            "Login returns JWT access + refresh tokens",
            "Token refresh endpoint",
            "Role CRUD (admin, editor, viewer)",
            "Permission-based middleware",
            "Password hashing with bcrypt",
            "Token blacklisting on logout",
            "Protected endpoints return 401/403 appropriately",
        ],
        tech_variations={"database": ["sqlite"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support OAuth2 authorization code flow",
            "Must implement rate limiting on login attempts",
            "Must support multi-factor authentication (simulated)",
        ],
        max_steps=200,
        expected_tasks=16,
    ),
    # ── Tier 5: Complex DB Operations ──
    SpecVariation(
        name="mongo_sql_engine",
        tier=5,
        base_description=(
            "Implement a query engine that provides SQL-style join operations "
            "on MongoDB collections. Support Inner, Left, Right, and Full Outer "
            "joins. First sync historical data from a simulated source, then "
            "transition to live updates via change stream simulation."
        ),
        acceptance_criteria=[
            "Inner join between two collections",
            "Left outer join",
            "Right outer join",
            "Full outer join",
            "Historical data sync from simulated source",
            "Live data updates via simulated change streams",
            "Query optimization for large datasets",
            "Proper error handling for disconnections",
        ],
        tech_variations={"database": ["mongodb_simulated"], "framework": ["fastapi"]},
        constraint_pool=[
            "Must support aggregation pipelines",
            "Must handle schema evolution",
        ],
        max_steps=300,
        expected_tasks=18,
    ),
]


class AdaptiveCurriculum:
    """Manages progressive difficulty scaling for RL training.

    Tracks agent performance and adaptively selects specification tiers.
    Promotes to harder tiers when performance exceeds threshold, and
    demotes when performance drops, implementing the Self-Improvement theme.
    """

    def __init__(
        self,
        promote_threshold: float = 0.7,
        demote_threshold: float = 0.3,
        window_size: int = 10,
        start_tier: int = 1,
        max_tier: int = 5,
    ):
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.window_size = window_size
        self.current_tier = start_tier
        self.max_tier = max_tier

        # Performance tracking per tier
        self._history: dict[int, deque] = {
            t: deque(maxlen=window_size) for t in range(1, max_tier + 1)
        }
        self._total_episodes = 0
        self._tier_episodes: dict[int, int] = {
            t: 0 for t in range(1, max_tier + 1)
        }

    def sample_spec(self) -> tuple[SpecVariation, str]:
        """Sample a specification from the current difficulty tier.

        Returns:
            Tuple of (spec_variation, rendered_description).
            The description includes randomized variations.
        """
        tier_specs = [s for s in SPEC_BANK if s.tier == self.current_tier]
        if not tier_specs:
            # Fallback to closest available tier
            tier_specs = [
                s
                for s in SPEC_BANK
                if s.tier == min(range(1, 6), key=lambda t: abs(t - self.current_tier))
            ]

        spec = random.choice(tier_specs)
        description = self._render_spec(spec)
        return spec, description

    def _render_spec(self, spec: SpecVariation) -> str:
        """Render a spec with randomized variations."""
        desc = spec.base_description

        # Add random tech variation
        for key, options in spec.tech_variations.items():
            chosen = random.choice(options)
            desc += f"\n\nTech choice - {key}: {chosen}"

        # Add random constraints (1-2 from pool)
        if spec.constraint_pool:
            num_constraints = min(
                random.randint(1, 2), len(spec.constraint_pool)
            )
            chosen_constraints = random.sample(
                spec.constraint_pool, num_constraints
            )
            desc += "\n\nAdditional requirements:"
            for c in chosen_constraints:
                desc += f"\n- {c}"

        return desc

    def record_episode(self, result: EpisodeResult):
        """Record an episode result and potentially adjust the tier."""
        tier = result.tier
        self._history[tier].append(result.success)
        self._total_episodes += 1
        self._tier_episodes[tier] = self._tier_episodes.get(tier, 0) + 1

        # Check for tier promotion
        if len(self._history[tier]) >= self.window_size:
            success_rate = sum(self._history[tier]) / len(self._history[tier])

            if (
                success_rate >= self.promote_threshold
                and self.current_tier < self.max_tier
            ):
                self.current_tier += 1

            elif (
                success_rate <= self.demote_threshold
                and self.current_tier > 1
            ):
                self.current_tier -= 1

    def get_stats(self) -> dict:
        """Return curriculum statistics."""
        stats = {
            "current_tier": self.current_tier,
            "total_episodes": self._total_episodes,
            "tier_episodes": dict(self._tier_episodes),
            "tier_success_rates": {},
        }
        for tier, history in self._history.items():
            if history:
                stats["tier_success_rates"][tier] = (
                    sum(history) / len(history)
                )
        return stats

    def get_max_steps_for_current_tier(self) -> int:
        """Return the maximum steps allowed for the current tier."""
        tier_specs = [s for s in SPEC_BANK if s.tier == self.current_tier]
        if tier_specs:
            return max(s.max_steps for s in tier_specs)
        return 100  # Default


class ReviewerPersonality(BaseModel):
    """Configuration for a simulated expert reviewer (Snorkel AI theme).

    Each personality has different coding preferences that may change
    during an episode, forcing the meta-agent to adapt.
    """

    name: str = "Default Reviewer"
    strictness: float = 0.5  # 0.0 (lenient) to 1.0 (strict)
    preferences: dict[str, str] = Field(default_factory=lambda: {
        "code_style": "functional",
        "typing": "strict",
        "error_handling": "comprehensive",
        "testing": "thorough",
        "documentation": "inline",
    })
    change_probability: float = 0.1  # Probability of preference change per step
    approval_threshold: float = 0.6  # Minimum quality to approve

    def maybe_change_preference(self) -> Optional[tuple[str, str, str]]:
        """Randomly change a preference with configured probability.

        Returns:
            Tuple of (key, old_value, new_value) if changed, else None.
        """
        if random.random() < self.change_probability:
            key = random.choice(list(self.preferences.keys()))
            old_value = self.preferences[key]
            alternatives = {
                "code_style": ["functional", "oop", "procedural"],
                "typing": ["strict", "relaxed", "gradual"],
                "error_handling": ["comprehensive", "minimal", "fail_fast"],
                "testing": ["thorough", "minimal", "property_based"],
                "documentation": ["inline", "docstring", "none"],
            }
            options = [
                v for v in alternatives.get(key, []) if v != old_value
            ]
            if options:
                new_value = random.choice(options)
                self.preferences[key] = new_value
                return (key, old_value, new_value)
        return None


# ── Pre-configured reviewer personalities ──

REVIEWER_POOL = [
    ReviewerPersonality(
        name="Senior Architect",
        strictness=0.8,
        preferences={
            "code_style": "oop",
            "typing": "strict",
            "error_handling": "comprehensive",
            "testing": "thorough",
            "documentation": "docstring",
        },
        change_probability=0.05,
        approval_threshold=0.7,
    ),
    ReviewerPersonality(
        name="Startup CTO",
        strictness=0.3,
        preferences={
            "code_style": "functional",
            "typing": "relaxed",
            "error_handling": "minimal",
            "testing": "minimal",
            "documentation": "none",
        },
        change_probability=0.15,
        approval_threshold=0.4,
    ),
    ReviewerPersonality(
        name="Security Lead",
        strictness=0.9,
        preferences={
            "code_style": "functional",
            "typing": "strict",
            "error_handling": "fail_fast",
            "testing": "thorough",
            "documentation": "inline",
        },
        change_probability=0.08,
        approval_threshold=0.8,
    ),
    ReviewerPersonality(
        name="ML Engineer",
        strictness=0.5,
        preferences={
            "code_style": "functional",
            "typing": "gradual",
            "error_handling": "comprehensive",
            "testing": "property_based",
            "documentation": "docstring",
        },
        change_probability=0.12,
        approval_threshold=0.5,
    ),
]
