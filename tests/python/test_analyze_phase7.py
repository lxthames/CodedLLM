import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase7.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase7", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def make_row(seed, strategy, p99_us, *, added=None, overhead=None, scenario="primary"):
    k = 16
    m = 8
    if added is None:
        added = {
            "wait": 0,
            "codedllm": m,
            "replication_capacity_normalized": m,
            "replication_full": k,
        }[strategy]
    if overhead is None:
        overhead = 100.0 * added / k
    return {
        "scenario": scenario,
        "seed": str(seed),
        "num_requests": "1000",
        "arrival_rate_hz": "2000.000",
        "k": str(k),
        "m": str(m),
        "degree": "3",
        "shard_size_mib": "16",
        "straggler_probability": "0.050",
        "queue_depth": "4",
        "gpu_concurrency": "2",
        "strategy": strategy,
        "added_shard_count": str(added),
        "storage_overhead_pct": str(overhead),
        "decode_cost_model": "staged_ready_to_result_cuda_benchmark_v1",
        "p50_us": str(p99_us // 2),
        "p95_us": str(p99_us - 10),
        "p99_us": str(p99_us),
        "recovery_admission_pct": "50.0" if strategy == "codedllm" else "0.0",
        "recovery_rejection_pct": "5.0" if strategy == "codedllm" else "0.0",
        "recovery_wait_pct": "30.0" if strategy == "codedllm" else "0.0",
        "recovery_unrecoverable_pct": "15.0" if strategy == "codedllm" else "0.0",
        "recovery_win_pct": "45.0" if strategy == "codedllm" else "0.0",
    }


class Phase7AnalysisTest(unittest.TestCase):
    def test_aggregates_paired_four_strategy_results(self):
        rows = []
        for seed, wait, capacity, full, coded in (
            (1, 100, 90, 70, 80),
            (2, 200, 120, 90, 100),
        ):
            rows.extend(
                [
                    make_row(seed, "wait", wait),
                    make_row(seed, "replication_capacity_normalized", capacity),
                    make_row(seed, "replication_full", full),
                    make_row(seed, "codedllm", coded),
                ]
            )

        summary = ANALYZER.analyze_rows(rows)
        coded = next(row for row in summary if row["strategy"] == "codedllm")

        self.assertEqual(coded["seed_count"], 2)
        self.assertAlmostEqual(coded["p99_mean_us"], 90.0)
        self.assertAlmostEqual(coded["p99_delta_vs_wait_mean_us"], 60.0)
        self.assertAlmostEqual(coded["p99_gain_vs_wait_mean_pct"], 35.0)
        self.assertEqual(coded["p99_win_vs_wait_seed_pct"], 100.0)
        self.assertEqual(
            coded["p99_win_vs_replication_capacity_normalized_seed_pct"], 100.0
        )
        self.assertEqual(coded["p99_win_vs_replication_full_seed_pct"], 0.0)
        self.assertGreater(coded["p99_ci95_us"], 0.0)

    def test_rejects_missing_pair_and_duplicate_pair(self):
        rows = [
            make_row(1, "wait", 100),
            make_row(1, "replication_capacity_normalized", 90),
            make_row(1, "codedllm", 80),
        ]
        with self.assertRaisesRegex(ValueError, "missing replication_full"):
            ANALYZER.analyze_rows(rows)

        rows.append(make_row(1, "replication_full", 70))
        rows.append(make_row(1, "replication_full", 71))
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            ANALYZER.analyze_rows(rows)

    def test_rejects_capacity_budget_mismatch_and_missing_calibration(self):
        rows = [
            make_row(1, "wait", 100),
            make_row(1, "replication_capacity_normalized", 90, added=7),
            make_row(1, "replication_full", 70),
            make_row(1, "codedllm", 80),
        ]
        with self.assertRaisesRegex(ValueError, "capacity-normalized storage"):
            ANALYZER.analyze_rows(rows)

        rows[1] = make_row(1, "replication_capacity_normalized", 90)
        del rows[0]["decode_cost_model"]
        with self.assertRaisesRegex(ValueError, "missing CSV columns"):
            ANALYZER.analyze_rows(rows)

    def test_conclusions_classify_positive_zero_and_negative_results(self):
        def summarize(wait, coded, scenario="primary"):
            return ANALYZER.analyze_rows(
                [
                    make_row(1, "wait", wait, scenario=scenario),
                    make_row(1, "replication_capacity_normalized", wait, scenario=scenario),
                    make_row(1, "replication_full", wait, scenario=scenario),
                    make_row(1, "codedllm", coded, scenario=scenario),
                ]
            )

        positive = ANALYZER.build_conclusions(summarize(100, 80))[0]
        zero = ANALYZER.build_conclusions(summarize(100, 100))[0]
        negative = ANALYZER.build_conclusions(summarize(100, 120, "queue_robustness"))[0]

        self.assertTrue(positive["wait_p99_interval_supported"])
        self.assertEqual(positive["conclusion_scope"], "primary_interval_supported")
        self.assertFalse(zero["wait_p99_interval_supported"])
        self.assertEqual(zero["conclusion_scope"], "not_supported")
        self.assertFalse(negative["wait_p99_interval_supported"])
        self.assertEqual(negative["conclusion_scope"], "exploratory")

        for row in (positive, zero, negative):
            self.assertIn("capacity_replication_p99_interval_supported", row)
            self.assertIn("full_replication_p99_interval_supported", row)
            self.assertIsInstance(row["capacity_replication_p99_interval_supported"], bool)
            self.assertIsInstance(row["full_replication_p99_interval_supported"], bool)


if __name__ == "__main__":
    unittest.main()
