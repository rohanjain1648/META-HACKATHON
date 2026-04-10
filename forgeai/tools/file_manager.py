"""File Manager — Safe file system operations with path validation.

Satisfies NFR-05: The framework must never execute shell commands that delete
files outside the project working directory.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from forgeai.core.activity_logger import ActivityLogger


class FileManager:
    """Safe file system manager that restricts all operations to the project directory."""

    def __init__(self, project_dir: str, logger: Optional[ActivityLogger] = None):
        self._project_dir = Path(project_dir).resolve()
        self._logger = logger
        self._files_created: list[str] = []
        self._files_modified: list[str] = []

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    def initialize_project(self):
        """Create the project directory if it doesn't exist."""
        self._project_dir.mkdir(parents=True, exist_ok=True)
        if self._logger:
            self._logger.info("FileManager", f"Project directory initialized: {self._project_dir}")

    def _validate_path(self, filepath: str) -> Path:
        """Ensure the path is within the project directory. Raises ValueError if not."""
        full_path = (self._project_dir / filepath).resolve()
        if not str(full_path).startswith(str(self._project_dir)):
            raise ValueError(
                f"SAFETY VIOLATION: Path '{filepath}' resolves outside project directory. "
                f"Resolved to: {full_path}, Project dir: {self._project_dir}"
            )
        return full_path

    def write_file(self, filepath: str, content: str) -> str:
        """Write content to a file within the project directory.
        
        Creates parent directories if needed. Returns the absolute path.
        """
        full_path = self._validate_path(filepath)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not full_path.exists()
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        rel = str(full_path.relative_to(self._project_dir))
        if is_new:
            self._files_created.append(rel)
        else:
            self._files_modified.append(rel)

        if self._logger:
            self._logger.file_write("FileManager", rel, {
                "action": "created" if is_new else "modified",
                "size_bytes": len(content),
                "lines": content.count("\n") + 1,
            })

        return str(full_path)

    def read_file(self, filepath: str) -> str:
        """Read a file from the project directory."""
        full_path = self._validate_path(filepath)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def file_exists(self, filepath: str) -> bool:
        """Check if a file exists in the project directory."""
        try:
            full_path = self._validate_path(filepath)
            return full_path.exists()
        except ValueError:
            return False

    def list_files(self, pattern: str = "**/*") -> list[str]:
        """List all files in the project directory matching a glob pattern."""
        files = []
        for p in self._project_dir.glob(pattern):
            if p.is_file():
                files.append(str(p.relative_to(self._project_dir)))
        return sorted(files)

    def create_directory(self, dirpath: str) -> str:
        """Create a directory within the project directory."""
        full_path = self._validate_path(dirpath)
        full_path.mkdir(parents=True, exist_ok=True)
        if self._logger:
            self._logger.info("FileManager", f"Created directory: {dirpath}")
        return str(full_path)

    def get_project_tree(self) -> dict:
        """Get the project directory tree as a nested dictionary."""
        tree = {}
        for filepath in self.list_files():
            parts = Path(filepath).parts
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = "file"
        return tree

    def get_all_source_files(self) -> dict[str, str]:
        """Read all Python source files and return {path: content} mapping."""
        result = {}
        for filepath in self.list_files("**/*.py"):
            try:
                result[filepath] = self.read_file(filepath)
            except Exception:
                pass
        return result

    def get_stats(self) -> dict:
        """Return file operation statistics."""
        return {
            "files_created": len(self._files_created),
            "files_modified": len(self._files_modified),
            "total_files": len(self.list_files()),
            "created_list": self._files_created,
            "modified_list": self._files_modified,
        }
