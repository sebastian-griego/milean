from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_results import (  # noqa: E402
    ArtifactSpec,
    FigureSpec,
    SlopeSpec,
    audit_repository,
    find_bootstrap,
    slope_from_mean_shift,
)


def artifact() -> dict:
    return {
        "config": {"baseline_accuracy": 0.75},
        "results": [
            {
                "layer": 3,
                "direction": "u",
                "mask": "global",
                "injection": "final",
                "alpha": -0.25,
                "mean_shift": -1.0,
                "n": 4,
            },
            {
                "layer": 3,
                "direction": "u",
                "mask": "global",
                "injection": "final",
                "alpha": 0.25,
                "mean_shift": 2.0,
                "n": 4,
            },
            {
                "layer": 3,
                "direction": "random",
                "mask": "global",
                "injection": "final",
                "alpha": -0.25,
                "mean_shift": -0.1,
                "n": 4,
            },
            {
                "layer": 3,
                "direction": "random",
                "mask": "global",
                "injection": "final",
                "alpha": 0.25,
                "mean_shift": 0.1,
                "n": 4,
            },
        ],
        "bootstrap": [
            {
                "layer": 3,
                "direction": "u",
                "mask": "global",
                "injection": "final",
                "alpha_neg": -0.25,
                "alpha_pos": 0.25,
                "slope": 6.0,
                "ci_low": 5.0,
                "ci_high": 7.0,
            },
            {
                "layer": 3,
                "direction": "random",
                "mask": "global",
                "injection": "final",
                "alpha_neg": -0.25,
                "alpha_pos": 0.25,
                "slope": 0.4,
                "ci_low": 0.2,
                "ci_high": 0.6,
            },
        ],
    }


class AuditResultsTest(unittest.TestCase):
    def test_slope_from_mean_shift_matches_bootstrap_key(self) -> None:
        data = artifact()
        row = find_bootstrap(data, layer=3, direction="u", mask="global", injection="final")

        self.assertEqual(slope_from_mean_shift(data, row), 6.0)

    def test_audit_repository_accepts_minimal_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "fixture.json").write_text(
                json.dumps(artifact()),
                encoding="utf-8",
            )
            figure = data_dir / "figure.png"
            figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)

            checks = audit_repository(
                root,
                artifact_specs=(
                    ArtifactSpec("data/fixture.json", expected_baseline=0.75, expected_n=4),
                ),
                slope_specs=(
                    SlopeSpec(
                        name="fixture slope",
                        path="data/fixture.json",
                        layer=3,
                        mask="global",
                        expected_slope=6.0,
                    ),
                ),
                figure_specs=(FigureSpec("data/figure.png", min_bytes=8),),
                control_artifact="data/fixture.json",
                control_slope_limit=0.5,
            )

            self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_audit_repository_flags_control_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            data_dir.mkdir()
            drifted = artifact()
            drifted["bootstrap"][1]["slope"] = 2.0
            (data_dir / "fixture.json").write_text(
                json.dumps(drifted),
                encoding="utf-8",
            )

            checks = audit_repository(
                root,
                artifact_specs=(
                    ArtifactSpec("data/fixture.json", expected_baseline=0.75, expected_n=4),
                ),
                slope_specs=(),
                figure_specs=(),
                control_artifact="data/fixture.json",
                control_slope_limit=0.5,
            )

            failures = [check for check in checks if not check["ok"]]
            self.assertEqual(len(failures), 1)
            self.assertIn("control slope", failures[0]["name"])


if __name__ == "__main__":
    unittest.main()
