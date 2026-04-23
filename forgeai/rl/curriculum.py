"""Curriculum Manager — Adaptive task difficulty for RL training.

The curriculum tracks per-task success rates and promotes the model to harder
tasks when it consistently succeeds, and demotes it when it consistently fails.
This ensures the model always operates near its capability frontier, keeping
reward signal non-zero (a key requirement from the hackathon guide).

Difficulty levels:
    EASY   (Level 1): Single pure functions, trivial edge cases
    MEDIUM (Level 2): Classes with multiple methods, state management
    HARD   (Level 3): Multi-function modules, algorithm challenges

Promotion threshold : success_rate > 0.70 over last 10 episodes
Demotion threshold  : success_rate < 0.25 over last 10 episodes
"""

import random
import textwrap
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class CodingTask:
    """A single RL training task."""
    task_id: str
    difficulty: DifficultyLevel
    description: str               # Natural language problem statement
    function_signature: str        # Expected function/class signature
    test_code: str                 # Pytest test suite (without the import)
    hint: str = ""                 # Optional warm-start hint
    canonical_solution: str = ""   # For SFT warm-up (not shown to model)


@dataclass
class CurriculumStats:
    recent_results: deque = field(default_factory=lambda: deque(maxlen=10))
    total_episodes: int = 0
    total_successes: int = 0

    @property
    def recent_success_rate(self) -> float:
        if not self.recent_results:
            return 0.5  # Default neutral
        return sum(self.recent_results) / len(self.recent_results)


class CurriculumManager:
    """Manages task sampling with adaptive difficulty."""

    PROMOTE_THRESHOLD = 0.70
    DEMOTE_THRESHOLD = 0.25

    def __init__(self, start_level: DifficultyLevel = DifficultyLevel.EASY):
        self._level = start_level
        self._stats = CurriculumStats()
        self._tasks: dict[DifficultyLevel, list[CodingTask]] = {
            DifficultyLevel.EASY: _build_easy_tasks(),
            DifficultyLevel.MEDIUM: _build_medium_tasks(),
            DifficultyLevel.HARD: _build_hard_tasks(),
        }

    @property
    def current_level(self) -> DifficultyLevel:
        return self._level

    @property
    def stats(self) -> CurriculumStats:
        return self._stats

    def sample_task(self) -> CodingTask:
        """Sample a task at the current difficulty level."""
        tasks = self._tasks[self._level]
        return random.choice(tasks)

    def record_result(self, success: bool, pass_rate: float) -> None:
        """Record episode result and potentially adjust difficulty."""
        self._stats.recent_results.append(1 if pass_rate >= 1.0 else 0)
        self._stats.total_episodes += 1
        if success:
            self._stats.total_successes += 1

        rate = self._stats.recent_success_rate
        # Only adjust after at least 5 episodes at current level
        if len(self._stats.recent_results) < 5:
            return

        if rate >= self.PROMOTE_THRESHOLD and self._level != DifficultyLevel.HARD:
            self._promote()
        elif rate <= self.DEMOTE_THRESHOLD and self._level != DifficultyLevel.EASY:
            self._demote()

    def _promote(self) -> None:
        levels = list(DifficultyLevel)
        idx = levels.index(self._level)
        if idx < len(levels) - 1:
            self._level = levels[idx + 1]
            self._stats.recent_results.clear()  # Reset window on promotion

    def _demote(self) -> None:
        levels = list(DifficultyLevel)
        idx = levels.index(self._level)
        if idx > 0:
            self._level = levels[idx - 1]
            self._stats.recent_results.clear()

    def get_all_tasks(self) -> list[CodingTask]:
        """Return all tasks across all levels (for SFT dataset generation)."""
        result = []
        for tasks in self._tasks.values():
            result.extend(tasks)
        return result


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _build_easy_tasks() -> list[CodingTask]:
    return [
        CodingTask(
            task_id="easy_001",
            difficulty=DifficultyLevel.EASY,
            description=(
                "Write a function `fibonacci(n: int) -> int` that returns the "
                "n-th Fibonacci number (0-indexed). fibonacci(0)=0, fibonacci(1)=1."
            ),
            function_signature="def fibonacci(n: int) -> int",
            test_code=textwrap.dedent("""\
                import pytest

                def test_base_cases():
                    assert fibonacci(0) == 0
                    assert fibonacci(1) == 1

                def test_small_values():
                    assert fibonacci(5) == 5
                    assert fibonacci(6) == 8
                    assert fibonacci(10) == 55

                def test_negative_raises():
                    with pytest.raises((ValueError, IndexError)):
                        fibonacci(-1)
            """),
            canonical_solution=textwrap.dedent("""\
                def fibonacci(n: int) -> int:
                    if n < 0:
                        raise ValueError("n must be non-negative")
                    if n <= 1:
                        return n
                    a, b = 0, 1
                    for _ in range(2, n + 1):
                        a, b = b, a + b
                    return b
            """),
        ),
        CodingTask(
            task_id="easy_002",
            difficulty=DifficultyLevel.EASY,
            description=(
                "Write a function `is_palindrome(s: str) -> bool` that returns "
                "True if the string is a palindrome (ignoring case and non-alphanumeric chars)."
            ),
            function_signature="def is_palindrome(s: str) -> bool",
            test_code=textwrap.dedent("""\
                def test_simple_palindrome():
                    assert is_palindrome("racecar") is True
                    assert is_palindrome("hello") is False

                def test_case_insensitive():
                    assert is_palindrome("Racecar") is True
                    assert is_palindrome("A man a plan a canal Panama") is True

                def test_empty_string():
                    assert is_palindrome("") is True

                def test_single_char():
                    assert is_palindrome("a") is True
            """),
            canonical_solution=textwrap.dedent("""\
                import re
                def is_palindrome(s: str) -> bool:
                    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
                    return cleaned == cleaned[::-1]
            """),
        ),
        CodingTask(
            task_id="easy_003",
            difficulty=DifficultyLevel.EASY,
            description=(
                "Write a function `count_words(text: str) -> dict[str, int]` "
                "that counts the frequency of each word (case-insensitive) in the text."
            ),
            function_signature="def count_words(text: str) -> dict",
            test_code=textwrap.dedent("""\
                def test_basic_counting():
                    result = count_words("hello world hello")
                    assert result["hello"] == 2
                    assert result["world"] == 1

                def test_case_insensitive():
                    result = count_words("Hello HELLO hello")
                    assert result["hello"] == 3

                def test_empty_string():
                    assert count_words("") == {}

                def test_punctuation_stripped():
                    result = count_words("hello, world!")
                    assert "hello" in result or "hello," in result
            """),
            canonical_solution=textwrap.dedent("""\
                import re
                from collections import Counter
                def count_words(text: str) -> dict:
                    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
                    return dict(Counter(words))
            """),
        ),
        CodingTask(
            task_id="easy_004",
            difficulty=DifficultyLevel.EASY,
            description=(
                "Write a function `two_sum(nums: list[int], target: int) -> tuple[int, int]` "
                "that returns indices of the two numbers that add up to target. "
                "Assume exactly one solution exists."
            ),
            function_signature="def two_sum(nums: list, target: int) -> tuple",
            test_code=textwrap.dedent("""\
                def test_basic():
                    i, j = two_sum([2, 7, 11, 15], 9)
                    assert sorted([i, j]) == [0, 1]

                def test_different_order():
                    i, j = two_sum([3, 2, 4], 6)
                    assert sorted([i, j]) == [1, 2]

                def test_same_element_twice():
                    i, j = two_sum([3, 3], 6)
                    assert sorted([i, j]) == [0, 1]
            """),
            canonical_solution=textwrap.dedent("""\
                def two_sum(nums: list, target: int) -> tuple:
                    seen = {}
                    for i, num in enumerate(nums):
                        complement = target - num
                        if complement in seen:
                            return (seen[complement], i)
                        seen[num] = i
                    raise ValueError("No solution found")
            """),
        ),
        CodingTask(
            task_id="easy_005",
            difficulty=DifficultyLevel.EASY,
            description=(
                "Write a function `flatten(nested: list) -> list` that flattens "
                "a nested list of arbitrary depth into a single flat list."
            ),
            function_signature="def flatten(nested: list) -> list",
            test_code=textwrap.dedent("""\
                def test_already_flat():
                    assert flatten([1, 2, 3]) == [1, 2, 3]

                def test_one_level():
                    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

                def test_deep_nesting():
                    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]

                def test_empty():
                    assert flatten([]) == []

                def test_mixed():
                    assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]
            """),
            canonical_solution=textwrap.dedent("""\
                def flatten(nested: list) -> list:
                    result = []
                    for item in nested:
                        if isinstance(item, list):
                            result.extend(flatten(item))
                        else:
                            result.append(item)
                    return result
            """),
        ),
    ]


def _build_medium_tasks() -> list[CodingTask]:
    return [
        CodingTask(
            task_id="medium_001",
            difficulty=DifficultyLevel.MEDIUM,
            description=(
                "Implement a `Stack` class with methods: `push(item)`, `pop() -> item`, "
                "`peek() -> item`, `is_empty() -> bool`, and `size() -> int`. "
                "Raise `IndexError` when pop/peek on empty stack."
            ),
            function_signature="class Stack",
            test_code=textwrap.dedent("""\
                import pytest

                def test_push_and_pop():
                    s = Stack()
                    s.push(1)
                    s.push(2)
                    assert s.pop() == 2
                    assert s.pop() == 1

                def test_peek():
                    s = Stack()
                    s.push(42)
                    assert s.peek() == 42
                    assert s.size() == 1  # peek doesn't remove

                def test_is_empty():
                    s = Stack()
                    assert s.is_empty() is True
                    s.push("x")
                    assert s.is_empty() is False

                def test_pop_empty_raises():
                    s = Stack()
                    with pytest.raises(IndexError):
                        s.pop()

                def test_peek_empty_raises():
                    s = Stack()
                    with pytest.raises(IndexError):
                        s.peek()

                def test_size():
                    s = Stack()
                    assert s.size() == 0
                    s.push(1); s.push(2); s.push(3)
                    assert s.size() == 3
            """),
            canonical_solution=textwrap.dedent("""\
                class Stack:
                    def __init__(self):
                        self._data = []

                    def push(self, item):
                        self._data.append(item)

                    def pop(self):
                        if self.is_empty():
                            raise IndexError("pop from empty stack")
                        return self._data.pop()

                    def peek(self):
                        if self.is_empty():
                            raise IndexError("peek from empty stack")
                        return self._data[-1]

                    def is_empty(self) -> bool:
                        return len(self._data) == 0

                    def size(self) -> int:
                        return len(self._data)
            """),
        ),
        CodingTask(
            task_id="medium_002",
            difficulty=DifficultyLevel.MEDIUM,
            description=(
                "Implement a `LRUCache` class with capacity parameter and methods: "
                "`get(key) -> int` (return -1 if not found) and `put(key, value)`. "
                "When capacity is exceeded, evict the least-recently used item."
            ),
            function_signature="class LRUCache",
            test_code=textwrap.dedent("""\
                def test_basic_get_put():
                    cache = LRUCache(2)
                    cache.put(1, 1)
                    cache.put(2, 2)
                    assert cache.get(1) == 1

                def test_eviction():
                    cache = LRUCache(2)
                    cache.put(1, 1)
                    cache.put(2, 2)
                    cache.put(3, 3)  # evicts key 1
                    assert cache.get(1) == -1
                    assert cache.get(3) == 3

                def test_lru_order():
                    cache = LRUCache(2)
                    cache.put(1, 1)
                    cache.put(2, 2)
                    cache.get(1)     # access 1, making 2 LRU
                    cache.put(3, 3)  # evicts 2
                    assert cache.get(2) == -1
                    assert cache.get(1) == 1

                def test_update_existing():
                    cache = LRUCache(1)
                    cache.put(1, 1)
                    cache.put(1, 10)
                    assert cache.get(1) == 10
            """),
            canonical_solution=textwrap.dedent("""\
                from collections import OrderedDict

                class LRUCache:
                    def __init__(self, capacity: int):
                        self.capacity = capacity
                        self._cache = OrderedDict()

                    def get(self, key: int) -> int:
                        if key not in self._cache:
                            return -1
                        self._cache.move_to_end(key)
                        return self._cache[key]

                    def put(self, key: int, value: int) -> None:
                        if key in self._cache:
                            self._cache.move_to_end(key)
                        self._cache[key] = value
                        if len(self._cache) > self.capacity:
                            self._cache.popitem(last=False)
            """),
        ),
        CodingTask(
            task_id="medium_003",
            difficulty=DifficultyLevel.MEDIUM,
            description=(
                "Write a function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                "that merges all overlapping intervals and returns them sorted."
            ),
            function_signature="def merge_intervals(intervals: list) -> list",
            test_code=textwrap.dedent("""\
                def test_basic_merge():
                    result = merge_intervals([[1,3],[2,6],[8,10],[15,18]])
                    assert result == [[1,6],[8,10],[15,18]]

                def test_no_overlap():
                    result = merge_intervals([[1,2],[3,4]])
                    assert result == [[1,2],[3,4]]

                def test_all_overlap():
                    result = merge_intervals([[1,4],[4,5]])
                    assert result == [[1,5]]

                def test_single():
                    result = merge_intervals([[1,1]])
                    assert result == [[1,1]]

                def test_empty():
                    assert merge_intervals([]) == []
            """),
            canonical_solution=textwrap.dedent("""\
                def merge_intervals(intervals: list) -> list:
                    if not intervals:
                        return []
                    intervals.sort(key=lambda x: x[0])
                    merged = [intervals[0]]
                    for start, end in intervals[1:]:
                        if start <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], end)
                        else:
                            merged.append([start, end])
                    return merged
            """),
        ),
    ]


def _build_hard_tasks() -> list[CodingTask]:
    return [
        CodingTask(
            task_id="hard_001",
            difficulty=DifficultyLevel.HARD,
            description=(
                "Implement a `Graph` class representing a directed weighted graph. "
                "Methods: `add_edge(src, dst, weight=1)`, "
                "`dijkstra(start) -> dict[node, int]` returning shortest distances from start, "
                "`has_cycle() -> bool`."
            ),
            function_signature="class Graph",
            test_code=textwrap.dedent("""\
                import pytest

                def test_dijkstra_simple():
                    g = Graph()
                    g.add_edge('A', 'B', 1)
                    g.add_edge('B', 'C', 2)
                    g.add_edge('A', 'C', 10)
                    dist = g.dijkstra('A')
                    assert dist['A'] == 0
                    assert dist['B'] == 1
                    assert dist['C'] == 3

                def test_has_cycle_true():
                    g = Graph()
                    g.add_edge(1, 2)
                    g.add_edge(2, 3)
                    g.add_edge(3, 1)
                    assert g.has_cycle() is True

                def test_has_cycle_false():
                    g = Graph()
                    g.add_edge(1, 2)
                    g.add_edge(2, 3)
                    assert g.has_cycle() is False

                def test_disconnected_dijkstra():
                    g = Graph()
                    g.add_edge('A', 'B', 5)
                    dist = g.dijkstra('A')
                    assert dist.get('C', float('inf')) == float('inf')
            """),
            canonical_solution=textwrap.dedent("""\
                import heapq
                from collections import defaultdict

                class Graph:
                    def __init__(self):
                        self._adj = defaultdict(list)
                        self._nodes = set()

                    def add_edge(self, src, dst, weight=1):
                        self._adj[src].append((dst, weight))
                        self._nodes.add(src)
                        self._nodes.add(dst)

                    def dijkstra(self, start):
                        dist = {node: float('inf') for node in self._nodes}
                        dist[start] = 0
                        heap = [(0, start)]
                        while heap:
                            d, u = heapq.heappop(heap)
                            if d > dist[u]:
                                continue
                            for v, w in self._adj[u]:
                                if dist[u] + w < dist[v]:
                                    dist[v] = dist[u] + w
                                    heapq.heappush(heap, (dist[v], v))
                        return dist

                    def has_cycle(self):
                        visited = set()
                        rec_stack = set()

                        def dfs(node):
                            visited.add(node)
                            rec_stack.add(node)
                            for neighbor, _ in self._adj[node]:
                                if neighbor not in visited:
                                    if dfs(neighbor):
                                        return True
                                elif neighbor in rec_stack:
                                    return True
                            rec_stack.discard(node)
                            return False

                        for node in list(self._nodes):
                            if node not in visited:
                                if dfs(node):
                                    return True
                        return False
            """),
        ),
        CodingTask(
            task_id="hard_002",
            difficulty=DifficultyLevel.HARD,
            description=(
                "Implement `TokenBucket` rate limiter class. "
                "Constructor: `TokenBucket(rate: float, capacity: float)` where rate is tokens/second. "
                "Method: `consume(tokens: float = 1.0) -> bool` returns True if request is allowed."
            ),
            function_signature="class TokenBucket",
            test_code=textwrap.dedent("""\
                import time
                import pytest

                def test_basic_allow():
                    tb = TokenBucket(rate=10, capacity=10)
                    assert tb.consume() is True

                def test_exceeds_capacity():
                    tb = TokenBucket(rate=1, capacity=3)
                    assert tb.consume(3) is True
                    assert tb.consume(1) is False  # bucket empty

                def test_refill_over_time():
                    tb = TokenBucket(rate=100, capacity=10)
                    tb.consume(10)  # drain
                    time.sleep(0.1)  # 100 tokens/s * 0.1s = 10 tokens refilled
                    assert tb.consume(5) is True

                def test_partial_consume():
                    tb = TokenBucket(rate=1, capacity=5)
                    assert tb.consume(3) is True
                    assert tb.consume(3) is False
                    assert tb.consume(2) is True
            """),
            canonical_solution=textwrap.dedent("""\
                import time

                class TokenBucket:
                    def __init__(self, rate: float, capacity: float):
                        self.rate = rate
                        self.capacity = capacity
                        self._tokens = capacity
                        self._last_refill = time.monotonic()

                    def _refill(self):
                        now = time.monotonic()
                        elapsed = now - self._last_refill
                        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                        self._last_refill = now

                    def consume(self, tokens: float = 1.0) -> bool:
                        self._refill()
                        if tokens <= self._tokens:
                            self._tokens -= tokens
                            return True
                        return False
            """),
        ),
    ]


