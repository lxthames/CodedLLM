#!/usr/bin/env python3
"""Aggregate Phase 7 seed runs and calculate deterministic 95% intervals."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


CONFIG_FIELDS = (
    "num_requests",
    "arrival_rate_hz",
    "k",
    "m",
    "degree",
    "shard_size_mib",
    "straggler_probability",
    "queue_depth",
    "gpu_concurrency",
    "strategy",
)
PAIR_FIELDS = CONFIG_FIELDS[:-1]
METRIC_FIELDS = (
    "p50_us",
    "p95_us",
    "p99_us",
    "recovery_admission_pct",
    "recovery_rejection_pct",
    "recovery_win_pct",
)
T_CRITICAL_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _mean_and_ci95(values):
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    degrees_of_freedom = len(values) - 1
    critical = (
        T_CRITICAL_95[degrees_of_freedom]
        if degrees_of_freedom < len(T_CRITICAL_95)
        else 1.96
    )
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def _config_key(row, fields):
    return tuple(row[field] for field in fields)


def _paired_gain(coded_rows, raw_index, baseline_strategy):
    gains = []
    wins = 0
    for coded in coded_rows:
        key = (coded["seed"], _config_key(coded, PAIR_FIELDS), baseline_strategy)
        baseline = raw_index.get(key)
        if baseline is None:
            raise ValueError(
                f"missing {baseline_strategy} pair for seed {coded['seed']}"
            )
        baseline_p99 = float(baseline["p99_us"])
        coded_p99 = float(coded["p99_us"])
        if baseline_p99 <= 0.0:
            raise ValueError("baseline P99 must be positive")
        gains.append(100.0 * (baseline_p99 - coded_p99) / baseline_p99)
        wins += coded_p99 < baseline_p99
    mean, ci95 = _mean_and_ci95(gains)
    return mean, ci95, 100.0 * wins / len(gains)


def analyze_rows(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Phase 7 input contains no rows")

    required = {"seed", *CONFIG_FIELDS, *METRIC_FIELDS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")

    groups = defaultdict(list)
    raw_index = {}
    for row in rows:
        groups[_config_key(row, CONFIG_FIELDS)].append(row)
        raw_index[
            (row["seed"], _config_key(row, PAIR_FIELDS), row["strategy"])
        ] = row

    summary = []
    for key in sorted(groups):
        group_rows = groups[key]
        seeds = {row["seed"] for row in group_rows}
        if len(seeds) != len(group_rows):
            raise ValueError("duplicate seed within one experiment configuration")

        result = dict(zip(CONFIG_FIELDS, key))
        result["seed_count"] = len(seeds)
        for metric in METRIC_FIELDS:
            mean, ci95 = _mean_and_ci95(
                [float(row[metric]) for row in group_rows]
            )
            output_name = metric.removesuffix("_us").removesuffix("_pct")
            unit = "_us" if metric.endswith("_us") else "_pct"
            result[f"{output_name}_mean{unit}"] = mean
            result[f"{output_name}_ci95{unit}"] = ci95

        if result["strategy"] == "codedllm":
            for baseline in ("wait", "replication"):
                mean, ci95, win_percent = _paired_gain(
                    group_rows, raw_index, baseline
                )
                result[f"p99_gain_vs_{baseline}_mean_pct"] = mean
                result[f"p99_gain_vs_{baseline}_ci95_pct"] = ci95
                result[f"p99_win_vs_{baseline}_seed_pct"] = win_percent
        summary.append(result)
    return summary


OUTPUT_FIELDS = (
    *CONFIG_FIELDS,
    "seed_count",
    "p50_mean_us",
    "p50_ci95_us",
    "p95_mean_us",
    "p95_ci95_us",
    "p99_mean_us",
    "p99_ci95_us",
    "recovery_admission_mean_pct",
    "recovery_admission_ci95_pct",
    "recovery_rejection_mean_pct",
    "recovery_rejection_ci95_pct",
    "recovery_win_mean_pct",
    "recovery_win_ci95_pct",
    "p99_gain_vs_wait_mean_pct",
    "p99_gain_vs_wait_ci95_pct",
    "p99_win_vs_wait_seed_pct",
    "p99_gain_vs_replication_mean_pct",
    "p99_gain_vs_replication_ci95_pct",
    "p99_win_vs_replication_seed_pct",
)


def _format_output(row):
    return {
        field: f"{row[field]:.6f}" if isinstance(row.get(field), float) else row.get(field, "")
        for field in OUTPUT_FIELDS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="raw multi-seed Phase 7 CSV")
    parser.add_argument("-o", "--output", help="summary CSV; defaults to stdout")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as source:
        summary = analyze_rows(csv.DictReader(source))

    destination = (
        open(args.output, "w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    try:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(_format_output(row) for row in summary)
    finally:
        if args.output:
            destination.close()


if __name__ == "__main__":
    main()
