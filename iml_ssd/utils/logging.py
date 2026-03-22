from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class CSVLogger:
    path: Path
    fieldnames: Optional[list] = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        # Add any missing keys
        for k in self.fieldnames:
            row.setdefault(k, None)
        self._writer.writerow(row)

    def flush(self) -> None:
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@dataclass
class RunLogger:
    run_dir: Path
    tb_subdir: str = "tb"

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.run_dir / self.tb_subdir
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.tb_dir))

        self.episode_csv = CSVLogger(self.run_dir / "episodes.csv")
        self.train_csv = CSVLogger(self.run_dir / "train.csv")

    def log_train(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": step, **metrics}
        self.train_csv.write(row)
        for k, v in metrics.items():
            try:
                self.tb.add_scalar(k, float(v), step)
            except Exception:
                pass

    def log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        row = {"episode": episode, **metrics}
        self.episode_csv.write(row)

    def save_config(self, cfg: Dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def close(self) -> None:
        try:
            self.tb.flush()
            self.tb.close()
        except Exception:
            pass
        self.episode_csv.close()
        self.train_csv.close()
