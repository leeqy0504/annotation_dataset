"""Manifest: track pipeline stage status and outputs."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path


class Manifest:
    """Tracks per-stage status, output paths, and timing."""

    def __init__(
        self,
        task: str,
        config_path: str,
        created_at: str | None = None,
        metadata: dict | None = None,
        run_id: str | None = None,
        status: str = "running",
        started_at: str | None = None,
        ended_at: str | None = None,
    ):
        self.task = task
        self.config_path = config_path
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.run_id = run_id
        self.status = status
        self.started_at = started_at or self.created_at
        self.ended_at = ended_at
        self.stages: dict[str, dict] = {}
        self.metadata = metadata or {}

    def mark_stage_done(
        self,
        name: str,
        output_dir: str,
        duration_s: float,
        metadata: dict | None = None,
    ):
        self.stages[name] = {
            "status": "done",
            "output_dir": output_dir,
            "duration_s": duration_s,
        }
        if metadata:
            self.stages[name]["metadata"] = copy.deepcopy(metadata)

    def mark_stage_failed(self, name: str):
        self.stages[name] = {
            "status": "failed",
            "output_dir": None,
            "duration_s": None,
        }
        self.status = "failed"
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def mark_stage_skipped(self, name: str):
        self.stages[name] = {
            "status": "skipped",
            "output_dir": None,
            "duration_s": 0,
        }

    def mark_completed(self):
        self.status = "completed"
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def mark_stopped(self):
        self.status = "stopped"
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def is_stage_done(self, name: str) -> bool:
        s = self.stages.get(name)
        return s is not None and s["status"] == "done"

    def get_output_dir(self, name: str) -> str | None:
        s = self.stages.get(name)
        if s:
            return s.get("output_dir")
        return None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "run_id": self.run_id,
            "status": self.status,
            "config_path": self.config_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "stages": self.stages,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path) as f:
            data = json.load(f)
        m = cls(
            task=data["task"],
            config_path=data["config_path"],
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
            run_id=data.get("run_id"),
            status=data.get("status", "running"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
        )
        m.stages = data.get("stages", {})
        return m

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def manifest_path_for_config(config) -> Path:
    """Return the manifest path for legacy or run-scoped pipeline configs."""
    base = Path(config.output_dir) / config.task
    if getattr(config, "run_id", None):
        return base / "runs" / config.run_id / "manifest.json"
    return base / "manifest.json"


def load_manifest_for_config(config) -> Manifest:
    path = manifest_path_for_config(config)
    if path.exists():
        return Manifest.load(str(path))
    return Manifest(
        task=config.task,
        config_path=getattr(config, "source_config_path", None) or config.output_dir,
        run_id=getattr(config, "run_id", None),
    )
