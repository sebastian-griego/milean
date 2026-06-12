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
    FigureSpec("data/plots_inencoder_rms_test200_goalcontext/steer_shift_layer12_block.png"),
)


def check_close(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
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


def slope_from_mean_shift(artifact: dict[str, Any], bootstrap_row: dict[str, Any]) -> float:
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
    return (float(row_pos["mean_shift"]) - float(row_neg["mean_shift"])) / (alpha_pos - alpha_neg)


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
            checks.append(check_condition(f"{spec.path} baseline accuracy", False, str(exc)))

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
            checks.append(check_condition(spec.name, False, f"{spec.path} was not loaded"))
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
            checks.append(check_condition(f"{spec.name} mean-shift consistency", False, str(exc)))

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


def render_report(checks: list[dict[str, Any]]) -> str:
    passed = sum(1 for check in checks if check["ok"])
    total = len(checks)
    lines = ["milean artifact audit", f"{passed}/{total} checks passed", ""]
    for check in checks:
        status = "PASS" if check["ok"] else "FAIL"
        lines.append(f"{status} {check['name']}: {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit committed milean result artifacts.")
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
    args = parser.parse_args(argv)

    checks = audit_repository(args.repo_root)
    print(render_report(checks))

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps({"checks": checks}, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.check and any(not check["ok"] for check in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
