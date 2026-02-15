#!/usr/bin/env python3
"""
Aggregate local GTrXL benchmark logs into a timestamped markdown report.

Expected runs:
- pendulum_r2d2_gtrxl_seed*
- pong_r2d2_gtrxl_seed*
- spaceinvaders_r2d2_gtrxl_seed*
- pendulum_vmpo_gtrxl_seed*
- pong_vmpo_gtrxl_seed*
- spaceinvaders_vmpo_gtrxl_seed*
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ENV_ORDER = ("pendulum", "pong", "spaceinvaders")
ALGO_ORDER = ("r2d2", "vmpo")

ENV_LABEL = {
    "pendulum": "Pendulum",
    "pong": "Pong",
    "spaceinvaders": "SpaceInvaders",
}

ALGO_LABEL = {
    "r2d2": "R2D2-GTrXL",
    "vmpo": "VMPO-GTrXL",
}

RUN_SPECS = (
    {"env": "pendulum", "algo": "r2d2", "prefix": "pendulum_r2d2_gtrxl_seed"},
    {"env": "pong", "algo": "r2d2", "prefix": "pong_r2d2_gtrxl_seed"},
    {"env": "spaceinvaders", "algo": "r2d2", "prefix": "spaceinvaders_r2d2_gtrxl_seed"},
    {"env": "pendulum", "algo": "vmpo", "prefix": "pendulum_vmpo_gtrxl_seed"},
    {"env": "pong", "algo": "vmpo", "prefix": "pong_vmpo_gtrxl_seed"},
    {"env": "spaceinvaders", "algo": "vmpo", "prefix": "spaceinvaders_vmpo_gtrxl_seed"},
)

LOG_RELATIVE_PATHS = (
    Path("log/evaluator/evaluator_logger.txt"),
    Path("log/evaluator_logger.txt"),
    Path("evaluator_logger.txt"),
)


@dataclass(frozen=True)
class RunLog:
    env: str
    algo: str
    prefix: str
    seed: Optional[int]
    path: Path
    points: List[Tuple[float, float]]

    @property
    def max_score(self) -> Optional[float]:
        if not self.points:
            return None
        return max(y for _, y in self.points)


@dataclass(frozen=True)
class AggregateCurve:
    x: List[float]
    mean: List[float]
    std: List[float]
    run_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GTrXL benchmark report from local logs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory where experiment folders/logs exist (default: current directory).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports"),
        help="Output report root directory (default: ./reports).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search nested directories recursively for matching experiment folders/log files.",
    )
    return parser.parse_args()


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_points(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    by_iter: Dict[float, float] = {}
    for train_iter, reward in points:
        if train_iter is None or reward is None:
            continue
        key = float(train_iter)
        if abs(key - round(key)) < 1e-9:
            key = float(int(round(key)))
        by_iter[key] = float(reward)
    return sorted(by_iter.items(), key=lambda x: x[0])


def extract_seed(path: Path) -> Optional[int]:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def discover_log_files(base_dir: Path, prefix: str, recursive: bool) -> List[Path]:
    found: Dict[str, Path] = {}

    def _maybe_add(candidate: Path) -> None:
        if candidate.is_file():
            found[str(candidate.resolve())] = candidate.resolve()

    for item in sorted(base_dir.glob(f"{prefix}*")):
        if item.is_dir():
            for rel in LOG_RELATIVE_PATHS:
                _maybe_add(item / rel)
        elif item.is_file():
            _maybe_add(item)

    for pattern in (f"{prefix}*.log", f"{prefix}*evaluator*.txt"):
        for item in sorted(base_dir.glob(pattern)):
            _maybe_add(item)

    if recursive:
        recursive_patterns = (
            f"**/{prefix}*/log/evaluator/evaluator_logger.txt",
            f"**/{prefix}*/log/evaluator_logger.txt",
            f"**/{prefix}*/evaluator_logger.txt",
            f"**/{prefix}*.log",
        )
        for pattern in recursive_patterns:
            for item in base_dir.glob(pattern):
                _maybe_add(item)

    return sorted(found.values())


def parse_evaluator_points(log_path: Path) -> List[Tuple[float, float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    points: List[Tuple[float, float]] = []
    parse_mode: Optional[str] = None
    current_train_iter: Optional[float] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("|") and "Name" in line and "train_iter" in line:
            parse_mode = "train_iter"
            continue
        if line.startswith("|") and "Name" in line and "reward_mean" in line:
            parse_mode = "reward_mean"
            continue

        if parse_mode and "|" in line and "Value" in line:
            pipe_start = line.find("|")
            row = line[pipe_start:]
            cols = [x.strip() for x in row.split("|")[1:-1]]
            if len(cols) >= 2 and cols[0] == "Value":
                value = _parse_float(cols[1])
                if value is not None:
                    if parse_mode == "train_iter":
                        current_train_iter = value
                    elif parse_mode == "reward_mean" and current_train_iter is not None:
                        points.append((current_train_iter, value))
            parse_mode = None

    if not points:
        fallback = re.compile(
            r"Train Iter\(([-+0-9.eE]+)\)\s*.*?Eval Return\(([-+0-9.eE]+)\)", re.IGNORECASE
        )
        for match in fallback.finditer(text):
            train_iter = _parse_float(match.group(1))
            reward = _parse_float(match.group(2))
            if train_iter is not None and reward is not None:
                points.append((train_iter, reward))

    return _normalize_points(points)


def aggregate_curves(runs: Sequence[RunLog]) -> AggregateCurve:
    if not runs:
        return AggregateCurve(x=[], mean=[], std=[], run_count=0)

    all_x = sorted({x for run in runs for x, _ in run.points})
    if not all_x:
        return AggregateCurve(x=[], mean=[], std=[], run_count=len(runs))

    run_xy = []
    for run in runs:
        xs = [x for x, _ in run.points]
        ys = [y for _, y in run.points]
        run_xy.append((xs, ys))

    means: List[float] = []
    stds: List[float] = []
    for x in all_x:
        values = []
        for xs, ys in run_xy:
            idx = bisect.bisect_right(xs, x) - 1
            if idx >= 0:
                values.append(ys[idx])
        if values:
            means.append(float(statistics.fmean(values)))
            stds.append(float(statistics.pstdev(values)) if len(values) > 1 else 0.0)
        else:
            means.append(float("nan"))
            stds.append(float("nan"))

    return AggregateCurve(x=all_x, mean=means, std=stds, run_count=len(runs))


def _format_score(score: Optional[float]) -> str:
    return "N/A" if score is None else f"{score:.3f}"


def create_plot(
    output_dir: Path,
    curves: Dict[str, Dict[str, AggregateCurve]],
) -> Tuple[Optional[Path], Optional[str]]:
    mpl_config_dir = output_dir / ".mplconfig"
    xdg_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency issue
        return None, f"Plot skipped: matplotlib unavailable ({exc})."

    fig, axes = plt.subplots(len(ENV_ORDER), 1, figsize=(10, 3.8 * len(ENV_ORDER)), squeeze=False)

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx][0]
        has_data = False
        for algo in ALGO_ORDER:
            curve = curves[env][algo]
            if not curve.x:
                continue
            has_data = True
            x = curve.x
            y = curve.mean
            s = curve.std
            label = f"{ALGO_LABEL[algo]} (n={curve.run_count})"
            ax.plot(x, y, label=label, linewidth=2)
            if curve.run_count > 1 and len(y) == len(s):
                lower = [yy - ss for yy, ss in zip(y, s)]
                upper = [yy + ss for yy, ss in zip(y, s)]
                ax.fill_between(x, lower, upper, alpha=0.15)

        ax.set_title(f"{ENV_LABEL[env]}: Reward Over Train Iteration")
        ax.set_ylabel("Reward mean")
        ax.grid(alpha=0.3, linestyle="--")
        if has_data:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No logs found", ha="center", va="center", transform=ax.transAxes)

    axes[-1][0].set_xlabel("Train iteration (time proxy)")
    fig.tight_layout()

    plot_path = output_dir / "score_over_time.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path, None


def build_readme(
    report_dir: Path,
    base_dir: Path,
    run_logs: Sequence[RunLog],
    curves: Dict[str, Dict[str, AggregateCurve]],
    max_scores: Dict[str, Dict[str, Optional[float]]],
    plot_path: Optional[Path],
    plot_note: Optional[str],
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# GTrXL Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{now_str}`")
    lines.append(f"- Base directory: `{base_dir.resolve()}`")
    lines.append("")
    lines.append("## Score Over Time")
    lines.append("")
    lines.append("X-axis uses train iteration as a time-weighted proxy.")
    lines.append("Each line is the seed-aggregated mean evaluator reward for one algorithm.")
    lines.append("")
    if plot_path is not None:
        lines.append(f"![Score over time]({plot_path.name})")
    if plot_note:
        lines.append("")
        lines.append(f"- {plot_note}")
    lines.append("")
    lines.append("## Max Score by Algorithm")
    lines.append("")
    lines.append("| Environment | R2D2-GTrXL | VMPO-GTrXL |")
    lines.append("| --- | ---: | ---: |")
    for env in ENV_ORDER:
        lines.append(
            f"| {ENV_LABEL[env]} | {_format_score(max_scores[env]['r2d2'])} | {_format_score(max_scores[env]['vmpo'])} |"
        )

    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Environment | Algorithm | Parsed Runs |")
    lines.append("| --- | --- | ---: |")
    for env in ENV_ORDER:
        for algo in ALGO_ORDER:
            parsed_runs = curves[env][algo].run_count
            lines.append(f"| {ENV_LABEL[env]} | {ALGO_LABEL[algo]} | {parsed_runs} |")

    lines.append("")
    lines.append("## Parsed Logs")
    lines.append("")
    if run_logs:
        for run in sorted(run_logs, key=lambda r: (r.env, r.algo, r.seed if r.seed is not None else -1, str(r.path))):
            seed_str = f"seed={run.seed}" if run.seed is not None else "seed=unknown"
            lines.append(
                f"- `{run.path.resolve()}` ({ENV_LABEL[run.env]}, {ALGO_LABEL[run.algo]}, {seed_str}, points={len(run.points)})"
            )
    else:
        lines.append("- No matching logs were parsed.")

    lines.append("")
    lines.append(f"_Report path: `{report_dir.resolve()}`_")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (base_dir / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)

    run_logs: List[RunLog] = []
    for spec in RUN_SPECS:
        env = spec["env"]
        algo = spec["algo"]
        prefix = spec["prefix"]
        for log_path in discover_log_files(base_dir, prefix, recursive=args.recursive):
            points = parse_evaluator_points(log_path)
            if not points:
                continue
            run_logs.append(
                RunLog(
                    env=env,
                    algo=algo,
                    prefix=prefix,
                    seed=extract_seed(log_path),
                    path=log_path,
                    points=points,
                )
            )

    grouped_runs: Dict[str, Dict[str, List[RunLog]]] = {env: {algo: [] for algo in ALGO_ORDER} for env in ENV_ORDER}
    for run in run_logs:
        grouped_runs[run.env][run.algo].append(run)

    curves: Dict[str, Dict[str, AggregateCurve]] = {env: {} for env in ENV_ORDER}
    max_scores: Dict[str, Dict[str, Optional[float]]] = {env: {} for env in ENV_ORDER}
    for env in ENV_ORDER:
        for algo in ALGO_ORDER:
            runs = grouped_runs[env][algo]
            curves[env][algo] = aggregate_curves(runs)
            max_value: Optional[float] = None
            for run in runs:
                if run.max_score is None:
                    continue
                if max_value is None or run.max_score > max_value:
                    max_value = run.max_score
            max_scores[env][algo] = max_value

    plot_path, plot_note = create_plot(report_dir, curves)
    readme_text = build_readme(report_dir, base_dir, run_logs, curves, max_scores, plot_path, plot_note)
    (report_dir / "README.md").write_text(readme_text, encoding="utf-8")

    json_summary = {
        "generated_at": datetime.now().isoformat(),
        "base_dir": str(base_dir),
        "report_dir": str(report_dir),
        "runs": [
            {
                "env": run.env,
                "algo": run.algo,
                "prefix": run.prefix,
                "seed": run.seed,
                "log_path": str(run.path),
                "points": [{"train_iter": x, "reward_mean": y} for x, y in run.points],
                "max_score": run.max_score,
            }
            for run in run_logs
        ],
        "max_scores": max_scores,
    }
    (report_dir / "summary.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Report created: {report_dir}")
    print(f"README: {report_dir / 'README.md'}")
    if plot_path is not None:
        print(f"Plot: {plot_path}")
    if plot_note:
        print(plot_note)


if __name__ == "__main__":
    main()
