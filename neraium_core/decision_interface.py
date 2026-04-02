from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional


def attach_action_signal_to_result(
    result: Dict[str, object],
    prior_action_signal: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Attach an action signal payload to a result and return it for next-frame reuse."""
    current = result.get("action_signal")
    if isinstance(current, dict):
        signal: Dict[str, object] = deepcopy(current)
    elif isinstance(prior_action_signal, dict):
        signal = deepcopy(prior_action_signal)
    else:
        signal = {"action": "hold", "confidence": 0.0}
    result["action_signal"] = signal
    return signal
