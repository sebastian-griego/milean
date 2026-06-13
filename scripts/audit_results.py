from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    expected_baseline: float
    expected_n: int
    baseline_tolerance: float = 1e-12


@dataclass(frozen=True)
class SlopeSpec:
    name: str
    path: str
    layer: int
    mask: str
    expected_slope: float
    direction: str = "u"
    injection: str = "final"
    tolerance: float = 1e-9
    expected_ci_low: float | None = None
    expected_ci_high: float | None = None
    ci_tolerance: float = 1e-9


@dataclass(frozen=True)
class FigureSpec:
    path: str
    min_bytes: int = 1024


ARTIFACT_SPECS = (
    ArtifactSpec(
        path="data/steer_sweep_test500.json",
        expected_baseline=0.798,
        expected_n=500,
    ),
    ArtifactSpec(
        path="data/steer_tokenloc_test500.json",
        expected_baseline=0.798,
        expected_n=500,
    ),
    ArtifactSpec(
        path="data/steer_inencoder_rms_test200_goalcontext.json",
        expected_baseline=0.805,
        expected_n=200,
    ),
)


SLOPE_SPECS = (
    SlopeSpec(
        name="control layer 11 global",
        path="data/steer_sweep_test500.json",
        layer=11,
        mask="global",
        expected_slope=9.36269833111763,
        expected_ci_low=8.857138985222578,
        expected_ci_high=9.834698380291464,
    ),
    SlopeSpec(
        name="control layer 12 global",
        path="data/steer_sweep_test500.json",
        layer=12,
        mask="global",
        expected_slope=17.731551934719086,
        expected_ci_low=16.81422329001427,
        expected_ci_high=18.693809566354755,
    ),
    SlopeSpec(
        name="token localized layer 12 global",
        path="data/steer_tokenloc_test500.json",
        layer=12,
        mask="global",
        expected_slope=17.731551934719086,
    ),
    SlopeSpec(
        name="token localized layer 12 goal",
        path="data/steer_tokenloc_test500.json",
        layer=12,
        mask="goal",
        expected_slope=7.672689042568207,
    ),
    SlopeSpec(
        name="token localized layer 12 context",
        path="data/steer_tokenloc_test500.json",
        layer=12,
        mask="context",
        expected_slope=10.179785448789596,
    ),
    SlopeSpec(
        name="token localized layer 12 goal_head",
        path="data/steer_tokenloc_test500.json",
        layer=12,
        mask="goal_head",
        expected_slope=7.136946946144104,
    ),
    SlopeSpec(
        name="in-encoder RMS layer 12 global",
        path="data/steer_inencoder_rms_test200_goalcontext.json",
        layer=12,
        mask="global",
        expected_slope=0.8232083868980408,
        injection="block",
    ),
    SlopeSpec(
        name="in-encoder RMS layer 12 goal",
        path="data/steer_inencoder_rms_test200_goalcontext.json",
        layer=12,
        mask="goal",
        expected_slope=0.37411234021186823,
        injection="block",
    ),
    SlopeSpec(
        name="in-encoder RMS layer 12 context",
        path="data/steer_inencoder_rms_test200_goalcontext.json",
        layer=12,
        mask="context",
        expected_slope=0.4493172860145569,
        injection="block",
    ),
)


FIGURE_SPECS = (
    FigureSpec("data/plots_test500/steer_shift_layer12.png"),
    FigureSpec("data/plots_tokenloc_test500/steer_shift_layer12_final.png"),
    FigureSpec(
        "data/plots_inencoder_rms_test200_goalcontext/steer_shift_layer12_block.png"
    ),
)


def check_close(
    name: str, actual: float, expected: float, tolerance: float
) -> dict[str, Any]:
    delta = abs(actual - expected)
    return {
        "name": name,
        "ok": delta <= tolerance,
        "actual": actual,
        "expected": expected,
        "tolerance": tolerance,
        "detail": f"actual={actual:.12g} expected={expected:.12g} delta={delta:.3g}",
    }


def check_condition(name: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": ok, "detail": detail}
    result.update(extra)
    return result


def load_artifact(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError(f"{relative_path} is not a JSON object")
    for key in ("config", "results", "bootstrap"):
        if key not in artifact:
            raise ValueError(f"{relative_path} is missing {key!r}")
    if not isinstance(artifact["config"], dict):
        raise ValueError(f"{relative_path} config is not an object")
    if not isinstance(artifact["results"], list):
        raise ValueError(f"{relative_path} results is not a list")
    if not isinstance(artifact["bootstrap"], list):
        raise ValueError(f"{relative_path} bootstrap is not a list")
    return artifact


def find_bootstrap(
    artifact: dict[str, Any],
    *,
    layer: int,
    direction: str,
    mask: str,
    injection: str,
) -> dict[str, Any]:
    for row in artifact["bootstrap"]:
        if (
            row.get("layer") == layer
            and row.get("direction") == direction
            and row.get("mask") == mask
            and row.get("injection") == injection
        ):
            return row
    raise ValueError(
        f"missing bootstrap row layer={layer} direction={direction} "
        f"mask={mask} injection={injection}"
    )


def find_result(
    artifact: dict[str, Any],
    *,
    layer: int,
    direction: str,
    mask: str,
    injection: str,
    alpha: float,
) -> dict[str, Any]:
    for row in artifact["results"]:
        if (
            row.get("layer") == layer
            and row.get("direction") == direction
            and row.get("mask") == mask
            and row.get("injection") == injection
            and abs(float(row.get("alpha")) - alpha) <= 1e-12
        ):
            return row
    raise ValueError(
        f"missing result row layer={layer} direction={direction} mask={mask} "
        f"injection={injection} alpha={alpha}"
    )


def slope_from_mean_shift(
    artifact: dict[str, Any], bootstrap_row: dict[str, Any]
) -> float:
    alpha_pos = float(bootstrap_row["alpha_pos"])
    alpha_neg = float(bootstrap_row["alpha_neg"])
    common = {
        "layer": int(bootstrap_row["layer"]),
        "direction": str(bootstrap_row["direction"]),
        "mask": str(bootstrap_row["mask"]),
        "injection": str(bootstrap_row["injection"]),
    }
    row_pos = find_result(artifact, alpha=alpha_pos, **common)
    row_neg = find_result(artifact, alpha=alpha_neg, **common)
    return (float(row_pos["mean_shift"]) - float(row_neg["mean_shift"])) / (
        alpha_pos - alpha_neg
    )


def png_digest(repo_root: Path, relative_path: str) -> tuple[int, str]:
    path = repo_root / relative_path
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{relative_path} is not a PNG file")
    return len(data), hashlib.sha256(data).hexdigest()


def audit_repository(
    repo_root: Path,
    *,
    artifact_specs: tuple[ArtifactSpec, ...] = ARTIFACT_SPECS,
    slope_specs: tuple[SlopeSpec, ...] = SLOPE_SPECS,
    figure_specs: tuple[FigureSpec, ...] = FIGURE_SPECS,
    control_artifact: str = "data/steer_sweep_test500.json",
    control_slope_limit: float = 1.25,
) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for spec in artifact_specs:
        try:
            artifact = load_artifact(repo_root, spec.path)
            artifacts[spec.path] = artifact
        except Exception as exc:
            checks.append(check_condition(f"load {spec.path}", False, str(exc)))
            continue

        try:
            checks.append(
                check_close(
                    f"{spec.path} baseline accuracy",
                    float(artifact["config"]["baseline_accuracy"]),
                    spec.expected_baseline,
                    spec.baseline_tolerance,
                )
            )
        except Exception as exc:
            checks.append(
                check_condition(f"{spec.path} baseline accuracy", False, str(exc))
            )

        try:
            row_counts = {int(row["n"]) for row in artifact["results"]}
            checks.append(
                check_condition(
                    f"{spec.path} row counts",
                    row_counts == {spec.expected_n},
                    f"observed_n={sorted(row_counts)} expected_n={spec.expected_n}",
                    observed=sorted(row_counts),
                    expected=spec.expected_n,
                )
            )
        except Exception as exc:
            checks.append(check_condition(f"{spec.path} row counts", False, str(exc)))

    for spec in slope_specs:
        artifact = artifacts.get(spec.path)
        if artifact is None:
            checks.append(
                check_condition(spec.name, False, f"{spec.path} was not loaded")
            )
            continue

        try:
            row = find_bootstrap(
                artifact,
                layer=spec.layer,
                direction=spec.direction,
                mask=spec.mask,
                injection=spec.injection,
            )
        except Exception as exc:
            checks.append(check_condition(spec.name, False, str(exc)))
            continue

        checks.append(
            check_close(
                f"{spec.name} slope",
                float(row["slope"]),
                spec.expected_slope,
                spec.tolerance,
            )
        )
        if spec.expected_ci_low is not None:
            checks.append(
                check_close(
                    f"{spec.name} ci_low",
                    float(row["ci_low"]),
                    spec.expected_ci_low,
                    spec.ci_tolerance,
                )
            )
        if spec.expected_ci_high is not None:
            checks.append(
                check_close(
                    f"{spec.name} ci_high",
                    float(row["ci_high"]),
                    spec.expected_ci_high,
                    spec.ci_tolerance,
                )
            )

        try:
            derived = slope_from_mean_shift(artifact, row)
            checks.append(
                check_close(
                    f"{spec.name} mean-shift consistency",
                    derived,
                    float(row["slope"]),
                    1e-6,
                )
            )
        except Exception as exc:
            checks.append(
                check_condition(f"{spec.name} mean-shift consistency", False, str(exc))
            )

    artifact = artifacts.get(control_artifact)
    if artifact is not None:
        for row in artifact["bootstrap"]:
            if row.get("direction") == "u":
                continue
            slope = float(row["slope"])
            name = (
                f"control slope layer {row.get('layer')} "
                f"{row.get('direction')} {row.get('mask')}"
            )
            checks.append(
                check_condition(
                    name,
                    abs(slope) <= control_slope_limit,
                    f"slope={slope:.12g} limit=+/-{control_slope_limit}",
                    actual=slope,
                    expected=f"abs(slope) <= {control_slope_limit}",
                )
            )

    for spec in figure_specs:
        try:
            size, digest = png_digest(repo_root, spec.path)
            checks.append(
                check_condition(
                    f"{spec.path} PNG",
                    size >= spec.min_bytes,
                    f"bytes={size} sha256={digest}",
                    bytes=size,
                    sha256=digest,
                    min_bytes=spec.min_bytes,
                )
            )
        except Exception as exc:
            checks.append(check_condition(f"{spec.path} PNG", False, str(exc)))

    return checks


def summarize_artifact(repo_root: Path, relative_path: str) -> dict[str, Any]:
    """Return compact, portfolio-friendly metrics for one steering artifact."""
    artifact = load_artifact(repo_root, relative_path)
    results = artifact["results"]
    bootstrap = artifact["bootstrap"]

    n_values = sorted({int(row["n"]) for row in results if "n" in row})
    alphas = sorted({float(row["alpha"]) for row in results if "alpha" in row})
    signal_rows = [row for row in bootstrap if row.get("direction") == "u"]
    control_rows = [row for row in bootstrap if row.get("direction") != "u"]
    signal_slopes = [_bootstrap_summary(row) for row in signal_rows]
    control_slopes = [_bootstrap_summary(row) for row in control_rows]
    max_signal = _max_abs(row["slope"] for row in signal_slopes)
    max_control = _max_abs(row["slope"] for row in control_slopes)

    return {
        "path": relative_path,
        "baseline_accuracy": float(artifact["config"].get("baseline_accuracy", 0.0)),
        "n_values": n_values,
        "layers": sorted({int(row["layer"]) for row in results if "layer" in row}),
        "masks": sorted({str(row["mask"]) for row in results if "mask" in row}),
        "injections": sorted(
            {str(row["injection"]) for row in results if "injection" in row}
        ),
        "alpha_min": min(alphas) if alphas else None,
        "alpha_max": max(alphas) if alphas else None,
        "signal_slopes": signal_slopes,
        "control_slopes": control_slopes,
        "max_abs_signal_slope": max_signal,
        "max_abs_control_slope": max_control,
        "signal_to_control_ratio": _ratio(max_signal, max_control),
    }


def build_metrics_summary(
    repo_root: Path,
    *,
    artifact_specs: tuple[ArtifactSpec, ...] = ARTIFACT_SPECS,
    slope_specs: tuple[SlopeSpec, ...] = SLOPE_SPECS,
    figure_specs: tuple[FigureSpec, ...] = FIGURE_SPECS,
) -> dict[str, Any]:
    """Build an auditable metrics summary from committed artifacts."""
    repo_root = repo_root.resolve()
    artifacts = [summarize_artifact(repo_root, spec.path) for spec in artifact_specs]
    figures = []
    for spec in figure_specs:
        size, digest = png_digest(repo_root, spec.path)
        figures.append(
            {
                "path": spec.path,
                "bytes": size,
                "sha256": digest,
                "min_bytes": spec.min_bytes,
            }
        )

    checks = audit_repository(
        repo_root,
        artifact_specs=artifact_specs,
        slope_specs=slope_specs,
        figure_specs=figure_specs,
    )
    passed = sum(1 for check in checks if check["ok"])
    return {
        "repo_root": str(repo_root),
        "artifacts": artifacts,
        "figures": figures,
        "audit_checks": {
            "passed": passed,
            "total": len(checks),
            "failed": len(checks) - passed,
        },
    }


def render_metrics_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# milean Artifact Metrics",
        "",
        f"- Audit checks: `{summary['audit_checks']['passed']}/{summary['audit_checks']['total']}` passed",
        "",
        "## Steering Artifacts",
        "",
        "| Artifact | N | Baseline | Signal slope | Max control | Signal/control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for artifact in summary["artifacts"]:
        n_values = ",".join(str(n) for n in artifact["n_values"])
        lines.append(
            "| "
            f"{artifact['path']} | {n_values} | "
            f"{artifact['baseline_accuracy']:.3f} | "
            f"{artifact['max_abs_signal_slope']:.3f} | "
            f"{artifact['max_abs_control_slope']:.3f} | "
            f"{_format_ratio(artifact['signal_to_control_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Key Slopes",
            "",
            "| Artifact | Layer | Mask | Injection | Slope | 95% CI |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for artifact in summary["artifacts"]:
        for row in artifact["signal_slopes"]:
            ci = _format_ci(row.get("ci_low"), row.get("ci_high"))
            lines.append(
                f"| {artifact['path']} | {row['layer']} | {row['mask']} | "
                f"{row['injection']} | {row['slope']:.3f} | {ci} |"
            )

    lines.extend(
        ["", "## Figures", "", "| Figure | Bytes | SHA256 |", "|---|---:|---|"]
    )
    for figure in summary["figures"]:
        lines.append(f"| {figure['path']} | {figure['bytes']} | `{figure['sha256']}` |")
    return "\n".join(lines) + "\n"


def render_report(checks: list[dict[str, Any]]) -> str:
    passed = sum(1 for check in checks if check["ok"])
    total = len(checks)
    lines = ["milean artifact audit", f"{passed}/{total} checks passed", ""]
    for check in checks:
        status = "PASS" if check["ok"] else "FAIL"
        lines.append(f"{status} {check['name']}: {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit committed milean result artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path for a JSON check summary.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when any audit check fails.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path for a JSON artifact metrics summary.",
    )
    parser.add_argument(
        "--metrics-md",
        type=Path,
        default=None,
        help="Optional path for a Markdown artifact metrics summary.",
    )
    args = parser.parse_args(argv)

    checks = audit_repository(args.repo_root)
    print(render_report(checks))

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps({"checks": checks}, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.metrics_json is not None or args.metrics_md is not None:
        metrics = build_metrics_summary(args.repo_root)
        if args.metrics_json is not None:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        if args.metrics_md is not None:
            args.metrics_md.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_md.write_text(
                render_metrics_markdown(metrics), encoding="utf-8"
            )

    if args.check and any(not check["ok"] for check in checks):
        return 1
    return 0


def _bootstrap_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "layer": int(row["layer"]),
        "direction": str(row["direction"]),
        "mask": str(row["mask"]),
        "injection": str(row["injection"]),
        "alpha_neg": float(row["alpha_neg"]),
        "alpha_pos": float(row["alpha_pos"]),
        "slope": float(row["slope"]),
    }
    if "ci_low" in row:
        summary["ci_low"] = float(row["ci_low"])
    if "ci_high" in row:
        summary["ci_high"] = float(row["ci_high"])
    if "reps" in row:
        summary["reps"] = int(row["reps"])
    return summary


def _max_abs(values: Any) -> float:
    values = [abs(float(value)) for value in values]
    return max(values) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None if numerator > 0 else 0.0
    return numerator / denominator


def _format_ratio(value: Any) -> str:
    if value is None:
        return "inf"
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.2f}"


def _format_ci(low: Any, high: Any) -> str:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return f"[{low:.3f}, {high:.3f}]"
    return "-"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
