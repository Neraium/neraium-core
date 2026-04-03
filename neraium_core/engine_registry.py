from __future__ import annotations

from pathlib import Path
from typing import Any

from neraium_core.alignment import StructuralEngine


class EngineRegistry:
    """Per-asset StructuralEngine registry with optional lifecycle metadata."""

    def __init__(
        self,
        engine_template: StructuralEngine,
        *,
        max_registry_size: int | None = None,
        track_last_access: bool = False,
        eviction_strategy: str | None = None,
    ) -> None:
        self.engine_template = engine_template
        self.max_registry_size = int(max_registry_size) if max_registry_size is not None else None
        self.track_last_access = bool(track_last_access)
        self.eviction_strategy = str(eviction_strategy or "none")
        self._engines_by_asset: dict[tuple[str, str, str], StructuralEngine] = {}
        self._last_access_seq = 0
        self._last_access_by_key: dict[tuple[str, str, str], int] = {}

    @property
    def engines(self) -> dict[tuple[str, str, str], StructuralEngine]:
        return self._engines_by_asset

    def clear(self) -> None:
        self._engines_by_asset.clear()
        self._last_access_by_key.clear()
        self._last_access_seq = 0

    def get_existing(self, key: tuple[str, str, str]) -> StructuralEngine | None:
        engine = self._engines_by_asset.get(key)
        if engine is not None:
            self._touch(key)
        return engine

    def get_or_create(
        self,
        frame: dict[str, Any],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> StructuralEngine:
        key = self._engine_key(frame)
        existing = self._engines_by_asset.get(key)
        if existing is not None:
            self._touch(key)
            return existing

        new_engine = self._build_engine_for_key(key, run_config=run_config)
        self._engines_by_asset[key] = new_engine
        self._touch(key)
        self._evict_if_needed()
        return new_engine

    def metadata_for_key(self, key: tuple[str, str, str]) -> dict[str, Any]:
        return {
            "key": key,
            "present": key in self._engines_by_asset,
            "last_access": self._last_access_by_key.get(key),
        }

    @staticmethod
    def _engine_key(frame: dict[str, Any]) -> tuple[str, str, str]:
        customer_id = str(frame.get("customer_id") or "default-customer")
        site_id = str(frame.get("site_id", "default-site"))
        asset_id = str(frame.get("asset_id", "default-asset"))
        return customer_id, site_id, asset_id

    def _build_engine_for_key(
        self,
        key: tuple[str, str, str],
        *,
        run_config: dict[str, Any] | None,
    ) -> StructuralEngine:
        customer_id, site_id, asset_id = key
        template = self.engine_template
        baseline_window = int(run_config.get("baseline_window", template.baseline_window)) if run_config else template.baseline_window
        recent_window = int(run_config.get("recent_window", template.recent_window)) if run_config else template.recent_window
        baseline_window = max(4, min(baseline_window, 500))
        recent_window = max(2, min(recent_window, 120))

        regime_path = "regime_library.json"
        try:
            base_path = Path(template.regime_store.path)
            safe_customer = customer_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_site = site_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_asset = asset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            regime_path = str(
                base_path.with_name(f"{base_path.stem}_{safe_customer}_{safe_site}_{safe_asset}{base_path.suffix}")
            )
        except Exception:
            regime_path = f"regime_library_{customer_id}_{site_id}_{asset_id}.json"

        frame_debug = bool((run_config or {}).get("frame_debug", getattr(template, "_frame_debug", False)))
        new_engine = StructuralEngine(
            baseline_window=baseline_window,
            recent_window=recent_window,
            window_stride=template.window_stride,
            regime_store_path=regime_path,
            baseline_adaptation_alpha=template.baseline_adaptation_alpha,
            representation_mode=(run_config or {}).get("representation_mode", template.representation_config.mode),
            reference_strategy=(run_config or {}).get("reference_strategy", template.representation_config.reference_strategy),
            context_diagnostics_enabled=bool((run_config or {}).get("context_diagnostics_enabled", template.representation_config.enable_diagnostics)),
            feature_weights=(run_config or {}).get(
                "feature_weights",
                {
                    "raw_weight": template.representation_config.weights.raw_weight,
                    "residual_weight": template.representation_config.weights.residual_weight,
                    "delta_weight": template.representation_config.weights.delta_weight,
                    "slope_weight": template.representation_config.weights.slope_weight,
                    "drift_weight": template.representation_config.weights.drift_weight,
                    "second_diff_weight": template.representation_config.weights.second_diff_weight,
                },
            ),
            frame_debug=frame_debug,
        )
        if run_config and run_config.get("baseline_locked") is True:
            new_engine.lock_baseline(True)
        return new_engine

    def _touch(self, key: tuple[str, str, str]) -> None:
        if not self.track_last_access:
            return
        self._last_access_seq += 1
        self._last_access_by_key[key] = self._last_access_seq

    def _evict_if_needed(self) -> None:
        if self.max_registry_size is None or self.max_registry_size <= 0:
            return
        if len(self._engines_by_asset) <= self.max_registry_size:
            return
        if self.eviction_strategy != "lru":
            return
        oldest_key = None
        oldest_seq = None
        for key in self._engines_by_asset.keys():
            seq = self._last_access_by_key.get(key, 0)
            if oldest_key is None or seq < (oldest_seq or seq):
                oldest_key = key
                oldest_seq = seq
        if oldest_key is not None:
            self._engines_by_asset.pop(oldest_key, None)
            self._last_access_by_key.pop(oldest_key, None)
