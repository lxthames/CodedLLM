import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase7.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase7", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def make_row(seed, strategy, p99_us):
    return {
        "seed": str(seed),
        "num_requests": "1000",
        "arrival_rate_hz": "2000.000",
        "k": "16",
        "m": "8",
        "degree": "3",
        "shard_size_mib": "16",
        "straggler_probability": "0.050",
        "queue_depth": "4",
        "gpu_concurrency": "2",
        "strategy": strategy,
        "p50_us": str(p99_us // 2),
        "p95_us": str(p99_us - 10),
        "p99_us": str(p99_us),
        "recovery_admission_pct": "50.0" if strategy == "codedllm" else "0.0",
        "recovery_rejection_pct": "5.0" if strategy == "codedllm" else "0.0",
        "recovery_win_pct": "45.0" if strategy == "codedllm" else "0.0",
    }


class Phase7AnalysisTest(unittest.TestCase):
    def test_aggregates_seeds_and_paired_p99_gains(self):
        rows = []
        for seed, wait, replication, coded in (
            (1, 100, 70, 80),
            (2, 200, 90, 100),
        ):
            rows.extend(
                [
                    make_row(seed, "wait", wait),
                    make_row(seed, "replication", replication),
                    make_row(seed, "codedllm", coded),
                ]
            )

        summary = ANALYZER.analyze_rows(rows)
        coded = next(row for row in summary if row["strategy"] == "codedllm")

        self.assertEqual(coded["seed_count"], 2)
        self.assertAlmostEqual(coded["p99_mean_us"], 90.0)
        self.assertAlmostEqual(coded["p99_gain_vs_wait_mean_pct"], 35.0)
        self.assertEqual(coded["p99_win_vs_wait_seed_pct"], 100.0)
        self.assertEqual(coded["p99_win_vs_replication_seed_pct"], 0.0)
        self.assertGreater(coded["p99_ci95_us"], 0.0)


if __name__ == "__main__":
    unittest.main()
