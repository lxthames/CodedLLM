import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "extract_phase7_decode_costs.py"
SPEC = importlib.util.spec_from_file_location("extract_phase7_decode_costs", MODULE_PATH)
EXTRACTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACTOR)


def benchmark(name, real_time=1.0, time_unit="us", **extra):
    return {
        "name": name,
        "real_time": real_time,
        "time_unit": time_unit,
        **extra,
    }


class Phase7DecodeCostExtractionTest(unittest.TestCase):
    def test_extracts_required_end_to_end_pairs_and_ignores_other_records(self):
        required = {(2, 1), (8, 16)}
        document = {
            "benchmarks": [
                benchmark("StagedReadyToResultLatency/2/1048576", 1001.1, "ns"),
                benchmark("StagedReadyToResultLatency/8/16777216", 1.001, "ms"),
                benchmark("StagedReadyToResultLatency/2/1048580", 99.0, "us"),
                benchmark("KernelOnlyLatency/2/1048576", 1.0, "ns"),
            ]
        }

        self.assertEqual(
            EXTRACTOR.extract_costs(document, required),
            {(2, 1): 2, (8, 16): 1001},
        )

    def test_rejects_missing_duplicate_and_error_records(self):
        required = {(2, 1)}

        with self.assertRaisesRegex(ValueError, "missing required"):
            EXTRACTOR.extract_costs({"benchmarks": []}, required)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            EXTRACTOR.extract_costs(
                {
                    "benchmarks": [
                        benchmark("StagedReadyToResultLatency/2/1048576"),
                        benchmark("suite/StagedReadyToResultLatency/2/1048576"),
                    ]
                },
                required,
            )

        with self.assertRaisesRegex(ValueError, "reported an error"):
            EXTRACTOR.extract_costs(
                {
                    "benchmarks": [
                        benchmark(
                            "StagedReadyToResultLatency/2/1048576",
                            error_occurred=True,
                        )
                    ]
                },
                required,
            )


if __name__ == "__main__":
    unittest.main()
