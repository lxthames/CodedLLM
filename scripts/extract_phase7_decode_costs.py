#!/usr/bin/env python3
"""Extract required end-to-end CUDA decode costs from Google Benchmark JSON."""

import argparse
import csv
import json
import math
import pathlib
import re


END_TO_END_NAME = re.compile(r"(?:^|/)EndToEndLatency/(\d+)/(\d+)(?:/|$)")
TIME_TO_MICROSECONDS = {
    "ns": 1.0e-3,
    "us": 1.0,
    "ms": 1.0e3,
    "s": 1.0e6,
}
DEFAULT_REQUIRED = "2:1,2:16,3:1,3:16,5:1,5:16,8:1,8:16"


def parse_required(value):
    required = set()
    for item in value.split(","):
        try:
            degree_text, shard_size_text = item.split(":", maxsplit=1)
            degree = int(degree_text)
            shard_size_mib = int(shard_size_text)
        except ValueError as error:
            raise ValueError("required pairs must use DEGREE:SIZE_MIB") from error
        if degree <= 0 or shard_size_mib <= 0:
            raise ValueError("required pairs must be positive")
        required.add((degree, shard_size_mib))
    if not required:
        raise ValueError("at least one required pair is needed")
    return required


def extract_costs(document, required):
    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("benchmark JSON does not contain benchmarks")

    costs = {}
    for benchmark in benchmarks:
        if benchmark.get("error_occurred") or benchmark.get("error_message"):
            raise ValueError(f"benchmark reported an error: {benchmark.get('name', '')}")
        match = END_TO_END_NAME.search(str(benchmark.get("name", "")))
        if not match:
            continue
        degree = int(match.group(1))
        shard_size_bytes = int(match.group(2))
        if shard_size_bytes % (1024 * 1024) != 0:
            continue
        key = (degree, shard_size_bytes // (1024 * 1024))
        if key not in required:
            continue
        time_unit = benchmark.get("time_unit")
        if time_unit not in TIME_TO_MICROSECONDS:
            raise ValueError(f"unsupported benchmark time unit: {time_unit}")
        try:
            real_time = float(benchmark["real_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing real_time for {benchmark.get('name', '')}") from error
        if not math.isfinite(real_time) or real_time <= 0.0:
            raise ValueError(f"invalid real_time for {benchmark.get('name', '')}")
        cost_us = math.ceil(real_time * TIME_TO_MICROSECONDS[time_unit])
        if key in costs:
            raise ValueError(f"duplicate end-to-end benchmark for {key[0]}:{key[1]}")
        costs[key] = cost_us

    missing = sorted(required.difference(costs))
    if missing:
        formatted = ", ".join(f"{degree}:{size}" for degree, size in missing)
        raise ValueError(f"missing required end-to-end benchmarks: {formatted}")
    return costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path, help="bench_decoder JSON")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--required", default=DEFAULT_REQUIRED,
                        help="comma-separated DEGREE:SIZE_MIB pairs")
    args = parser.parse_args()

    required = parse_required(args.required)
    with args.input.open(encoding="utf-8") as source:
        costs = extract_costs(json.load(source), required)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("degree", "shard_size_mib", "end_to_end_us"))
        for degree, shard_size_mib in sorted(costs):
            writer.writerow((degree, shard_size_mib, costs[(degree, shard_size_mib)]))


if __name__ == "__main__":
    main()
