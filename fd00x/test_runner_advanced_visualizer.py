"""Advanced visualizations for multi-condition FD004 analysis.

Shows:
- Component behavior across 6 operating conditions
- Fault mode signature patterns
- Per-condition lead times
- Degradation trajectory classification
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .config import DetectorConfig
from .detector import StructuralDriftDetector, UnitScores
from .evaluation import load_cmapss_dataset


class AdvancedFD004Visualizer:
    """Visualizations for multi-condition FD004 analysis."""

    def __init__(self):
        self.condition_colors = {
            1: "#FF6B6B",
            2: "#4ECDC4",
            3: "#45B7D1",
            4: "#FFA07A",
            5: "#98D8C8",
            6: "#FFB6C1",
        }
        self.fault_colors = {
            "HPC_like": "#FF6B6B",
            "Fan_like": "#4ECDC4",
            "mixed": "#FFD700",
            "unknown": "#999999",
        }

    def plot_condition_comparison(
        self,
        results: Dict[int, Tuple[str, List[float]]],  # condition_id -> (component, lead_times)
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Compare component activation across 6 operating conditions.

        Args:
            results: Dict mapping condition_id to (dominant_component, [lead_times])
            output_path: Save path or None

        Returns:
            Path to saved figure or None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        conditions = sorted(results.keys())
        components = ["quantum", "information", "free_energy", "topological", "algorithmic"]

        # Panel 1: Component activation by condition
        condition_comp_counts = {i: {c: 0 for c in components} for i in conditions}
        for cond_id, (dom_comp, _) in results.items():
            if dom_comp in components:
                condition_comp_counts[cond_id][dom_comp] += 1

        comp_data = {c: [condition_comp_counts[cond].get(c, 0) for cond in conditions] for c in components}

        x = np.arange(len(conditions))
        width = 0.15
        colors_map = {
            "quantum": "#FF6B6B",
            "information": "#4ECDC4",
            "free_energy": "#45B7D1",
            "topological": "#FFA07A",
            "algorithmic": "#98D8C8",
        }

        for i, component in enumerate(components):
            offset = (i - 2) * width
            ax1.bar(x + offset, comp_data[component], width, label=component, color=colors_map[component], alpha=0.8)

        ax1.set_xlabel("Operating Condition", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Units with Dominant Component", fontsize=11, fontweight="bold")
        ax1.set_title("Component Dominance Across 6 Operating Conditions", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Cond {i}" for i in conditions])
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Lead time by condition
        condition_leads = {i: [] for i in conditions}
        for cond_id, (_, leads) in results.items():
            condition_leads[cond_id] = [x for x in leads if x is not None and x > 0]

        lead_medians = []
        lead_errors = []
        for cond_id in conditions:
            leads = condition_leads[cond_id]
            if leads:
                lead_medians.append(np.median(leads))
                q1, q3 = np.percentile(leads, [25, 75])
                lead_errors.append([np.median(leads) - q1, q3 - np.median(leads)])
            else:
                lead_medians.append(0)
                lead_errors.append([0, 0])

        errors = np.array(lead_errors).T if lead_errors else None
        ax2.bar(x, lead_medians, color=[self.condition_colors[i] for i in conditions], alpha=0.7)
        if errors is not None and errors.shape[1] > 0:
            ax2.errorbar(x, lead_medians, yerr=errors, fmt="none", color="black", capsize=5)

        ax2.set_xlabel("Operating Condition", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Median Lead Time (cycles)", fontsize=11, fontweight="bold")
        ax2.set_title("Lead Time by Operating Condition", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Cond {i}" for i in conditions])
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            return None

    def plot_fault_mode_distribution(
        self,
        fault_modes: Dict[str, int],
        component_by_mode: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Visualize fault mode distribution and component signatures.

        Args:
            fault_modes: Dict mapping fault_mode -> count
            component_by_mode: Dict mapping fault_mode -> {component -> activation_rate}
            output_path: Save path or None

        Returns:
            Path to saved figure or None
        """
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.3)

        # Panel 1: Fault mode pie chart
        ax1 = fig.add_subplot(gs[0])
        modes = list(fault_modes.keys())
        counts = list(fault_modes.values())
        colors = [self.fault_colors.get(m, "#999999") for m in modes]

        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=modes,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)
        ax1.set_title("Inferred Fault Mode Distribution\n(2 modes across 249 units)", fontsize=12, fontweight="bold")

        # Panel 2: Component signatures per fault mode
        ax2 = fig.add_subplot(gs[1])

        components = ["quantum", "information", "free_energy", "topological", "algorithmic"]
        x = np.arange(len(components))
        width = 0.25

        colors_map = {
            "quantum": "#FF6B6B",
            "information": "#4ECDC4",
            "free_energy": "#45B7D1",
            "topological": "#FFA07A",
            "algorithmic": "#98D8C8",
        }

        mode_list = ["HPC_like", "Fan_like", "mixed"]
        for i, mode in enumerate(mode_list):
            if mode in component_by_mode:
                values = [component_by_mode[mode].get(c, 0) for c in components]
                offset = (i - 1) * width
                ax2.bar(x + offset, values, width, label=mode, alpha=0.8)

        ax2.set_xlabel("Intelligence Component", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Activation Rate in Fault Mode", fontsize=11, fontweight="bold")
        ax2.set_title("Component Signatures per Fault Mode", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(components, rotation=45, ha="right")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim([0, 1])

        plt.suptitle("FD004: Fault Mode Analysis (HPC vs Fan Degradation)", fontsize=13, fontweight="bold", y=1.00)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            return None

    def plot_trajectory_classification(
        self,
        trajectories: Dict[str, int],  # "smooth" -> count, "stepped" -> count, etc.
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Visualize degradation trajectory classification.

        Args:
            trajectories: Dict mapping trajectory_type -> count
            output_path: Save path or None

        Returns:
            Path to saved figure or None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Trajectory distribution
        traj_types = list(trajectories.keys())
        counts = list(trajectories.values())
        colors = ["#45B7D1", "#FFA07A", "#FF6B6B"]

        ax1.barh(traj_types, counts, color=colors[:len(traj_types)], alpha=0.7)
        ax1.set_xlabel("Number of Units", fontsize=11, fontweight="bold")
        ax1.set_title("Degradation Trajectory Classification", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # Add percentages
        total = sum(counts)
        for i, (traj, count) in enumerate(zip(traj_types, counts)):
            pct = (count / total) * 100
            ax1.text(count + 2, i, f"{count} ({pct:.1f}%)", va="center", fontsize=10)

        # Panel 2: Trajectory patterns illustration
        ax2.axis("off")

        # Draw example trajectories
        x = np.linspace(0, 1, 50)

        # Smooth trajectory
        y_smooth = x ** 1.5
        ax2.plot(x, y_smooth, linewidth=3, color="#45B7D1", label="Smooth (monotonic decline)")

        # Stepped trajectory
        y_stepped = np.where(x < 0.5, x * 0.3, x * 0.3 + 0.35)
        ax2.plot(x, y_stepped, linewidth=3, color="#FFA07A", label="Stepped (sudden jumps)", linestyle="--")

        # Variable trajectory
        y_var = x + 0.2 * np.sin(10 * x)
        ax2.plot(x, y_var, linewidth=3, color="#FF6B6B", label="Variable (high noise)", linestyle=":")

        ax2.set_xlabel("Normalized Time →", fontsize=10)
        ax2.set_ylabel("Normalized Drift Signal →", fontsize=10)
        ax2.set_title("Degradation Pattern Types", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=10)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.2])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            return None

    def plot_condition_unit_lifecycle(
        self,
        unit_id: int,
        operating_conditions: np.ndarray,
        ema_drift: np.ndarray,
        warning_index: Optional[int] = None,
        component_dominance: Optional[Dict[int, str]] = None,
        output_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Show how unit moves through operating conditions and drift levels.

        Args:
            unit_id: Unit identifier
            operating_conditions: Array of condition IDs per cycle
            ema_drift: EMA drift score per cycle
            warning_index: Alert cycle or None
            component_dominance: Dict mapping condition_id -> dominant_component
            output_path: Save path or None

        Returns:
            Path to saved figure or None
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        cycles = np.arange(len(operating_conditions))

        # Panel 1: Operating conditions over time
        ax1 = axes[0]
        for cond_id in range(1, 7):
            mask = operating_conditions == cond_id
            ax1.fill_between(
                cycles, 0, 1, where=mask,
                alpha=0.3,
                color=self.condition_colors[cond_id],
                label=f"Condition {cond_id}"
            )

        ax1.set_ylabel("Operating Condition", fontsize=11, fontweight="bold")
        ax1.set_title(f"Unit {unit_id}: Operating Condition Lifecycle", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", ncol=3, fontsize=9)
        ax1.set_ylim([0, 1])
        ax1.set_yticks([])

        # Panel 2: Drift signal with condition coloring
        ax2 = axes[1]

        # Plot drift with background colored by condition
        for cond_id in range(1, 7):
            mask = operating_conditions == cond_id
            if np.any(mask):
                ax2.fill_between(
                    cycles[mask], 0, ema_drift[mask],
                    alpha=0.2,
                    color=self.condition_colors[cond_id]
                )

        ax2.plot(cycles, ema_drift, linewidth=2.5, color="darkblue", label="EMA Drift Score")

        if warning_index is not None:
            ax2.axvline(warning_index, color="red", linestyle="--", linewidth=2, label=f"Alert (cycle {warning_index})")
            ax2.scatter([warning_index], [ema_drift[warning_index]], color="red", s=100, zorder=5)

        ax2.set_ylabel("EMA Drift Score", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Cycle", fontsize=11, fontweight="bold")
        ax2.set_title("Drift Trajectory Across Conditions", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            return None
