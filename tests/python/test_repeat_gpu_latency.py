import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'repeat_gpu_latency', Path(__file__).resolve().parents[2] / 'scripts/repeat_gpu_latency.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepeatedLatencyTest(unittest.TestCase):
    def test_constant_interval(self):
        self.assertEqual(MODULE.interval([3.0] * 30)['ci95_half_width'], 0)

    def test_known_interval(self):
        result = MODULE.interval(list(range(30)))
        self.assertEqual(result['mean'], 14.5)
        self.assertAlmostEqual(result['ci95_half_width'], MODULE.T29 * math.sqrt(77.5 / 30))

    def test_incomplete_and_nonfinite_rejected(self):
        for values in ([1] * 29, [1] * 29 + [float('nan')]):
            with self.assertRaises(ValueError):
                MODULE.interval(values)

    def test_extraction_units(self):
        row = dict(name='SparseDecoderBenchmark/EndToEndLatency/3/16777216',
                   iterations=20, real_time=1230000, time_unit='ns')
        self.assertEqual(MODULE.extract({'benchmarks': [row]}, 'EndToEndLatency', 3), 1.23)

    def test_error_and_iteration_validation(self):
        row = dict(name='SparseDecoderBenchmark/KernelOnlyLatency/2/16777216/iterations:100/manual_time',
                   iterations=20, real_time=0.1, time_unit='ms')
        with self.assertRaises(ValueError):
            MODULE.extract({'benchmarks': [row]}, 'KernelOnlyLatency', 2)
        row.update(iterations=100, error_occurred=True)
        with self.assertRaises(ValueError):
            MODULE.extract({'benchmarks': [row]}, 'KernelOnlyLatency', 2)


if __name__ == '__main__':
    unittest.main()
