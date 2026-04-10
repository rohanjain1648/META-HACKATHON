"""Activity Logger — Timestamped, append-only log of every framework action.

Satisfies NFR-02: Every action (API call, file write, test run) must be logged
in a human-readable activity log.
"""

import json
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    AGENT = "AGENT"
    CHECKPOINT = "CHECKPOINT"
    API_CALL = "API_CALL"
    FILE_WRITE = "FILE_WRITE"
    TEST_RUN = "TEST_RUN"


class LogEntry(BaseModel):
    timestamp: str
    level: LogLevel
    source: str
    message: str
    details: Optional[dict] = None


class ActivityLogger:
    """Thread-safe, append-only activity logger with both file and in-memory output."""

    def __init__(self, log_file: str = "./forgeai_activity.log"):
        self._log_file = Path(log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[LogEntry] = []
        self._lock = threading.Lock()
        self._listeners: list[Any] = []  # Callbacks for real-time streaming

        # Write header
        self._write_to_file(
            f"\n{'='*80}\n"
            f"ForgeAI Activity Log — Started {datetime.now().isoformat()}\n"
            f"{'='*80}\n"
        )

    def add_listener(self, callback):
        """Register a callback for real-time log streaming (used by web dashboard)."""
        self._listeners.append(callback)

    def remove_listener(self, callback):
        """Unregister a log listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def log(self, level: LogLevel, source: str, message: str, details: Optional[dict] = None):
        """Append a timestamped log entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            source=source,
            message=message,
            details=details,
        )
        
        with self._lock:
            self._entries.append(entry)
        
        # Write to file
        detail_str = f" | {json.dumps(details)}" if details else ""
        line = f"[{entry.timestamp}] [{level.value:>10}] [{source}] {message}{detail_str}"
        self._write_to_file(line + "\n")

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(entry)
            except Exception:
                pass

    def info(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.INFO, source, message, details)

    def warn(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.WARN, source, message, details)

    def error(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.ERROR, source, message, details)

    def agent(self, agent_name: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.AGENT, agent_name, message, details)

    def api_call(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.API_CALL, source, message, details)

    def file_write(self, source: str, filepath: str, details: Optional[dict] = None):
        self.log(LogLevel.FILE_WRITE, source, f"Wrote file: {filepath}", details)

    def test_run(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.TEST_RUN, source, message, details)

    def checkpoint(self, source: str, message: str, details: Optional[dict] = None):
        self.log(LogLevel.CHECKPOINT, source, message, details)

    def get_entries(self, level: Optional[LogLevel] = None, limit: int = 100) -> list[LogEntry]:
        """Return recent log entries, optionally filtered by level."""
        with self._lock:
            entries = self._entries
            if level:
                entries = [e for e in entries if e.level == level]
            return entries[-limit:]

    def get_all_entries(self) -> list[LogEntry]:
        """Return all log entries."""
        with self._lock:
            return list(self._entries)

    def _write_to_file(self, content: str):
        """Append content to the log file."""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass  # Logging should never crash the framework
