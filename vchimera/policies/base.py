from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..protocol import CommsAction


class BasePolicy:
    name: str = "base"

    def reset(self) -> None:
        return

    def act(self, obs: Dict, t: int) -> Tuple[str, CommsAction]:
        raise NotImplementedError


@dataclass
class PolicyOutput:
    cyber_action: str
    comms_action: CommsAction
