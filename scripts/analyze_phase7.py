#!/usr/bin/env python3
"""Aggregate paired Phase 7 experiments and report confidence-bound conclusions."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


SCENARIO_FIELDS = (
    "scenario",
    "num_requests",
    "arrival_rate_hz",
    "k",
    "m",
    "degree",
    "shard_size_mib",
    "straggler_probability",
    "queue_depth",
    "gpu_concurrency",
    "decode_cost_model",
)
BUDGET_FIELDS = ("added_shard_count", "storage_overhead_pct")
CONFIG_FIELDS = (*SCENARIO_FIELDS, "strategy", *BUDGET_FIELDS)
PAIR_FIELDS = SCENARIO_FIELDS
METRIC_FIELDS = (
    "p50_us",
    "p95_us",
    "p99_us",
    "recovery_admission_pct",
    "recovery_rejection_pct",
    "recovery_wait_pct",
    "recovery_unrecoverable_pct",
    "recovery_win_pct",
)
BASELINES = (
    "wait",
    "replication_capacity_normalized",
    "replication_full",
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


def _float(row, field):
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"invalid {field}")
    return value


def _validate_budget(row):
    strategy = row["strategy"]
    k = _float(row, "k")
    m = _float(row, "m")
    added = _float(row, "added_shard_count")
    overhead = _float(row, "storage_overhead_pct")
    if k <= 0.0 or m <= 0.0 or added < 0.0 or overhead < 0.0:
        raise ValueError("invalid storage budget")
    if not row["decode_cost_model"]:
        raise ValueError("missing decode cost model")
    expected_overhead = 100.0 * added / k
    if not math.isclose(overhead, expected_overhead, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("storage overhead does not match added shard count")
    if strategy == "wait" and (added != 0.0 or overhead != 0.0):
        raise ValueError("wait baseline must not add storage")
    if strategy in ("codedllm", "replication_capacity_normalized"):
        if not math.isclose(added, m, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("capacity-normalized storage must add m shards")
    if strategy == "replication_full" and not math.isclose(
        added, k, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("full replication must add k shards")


def _paired_improvement(coded_rows, raw_index, baseline_strategy):
    absolute_deltas = []
    percentage_gains = []
    wins = 0
    for coded in coded_rows:
        key = (coded["seed"], _config_key(coded, PAIR_FIELDS), baseline_strategy)
        baseline = raw_index.get(key)
        if baseline is None:
            raise ValueError(
                f"missing {baseline_strategy} pair for seed {coded['seed']}"
            )
        if baseline_strategy == "replication_capacity_normalized":
            for field in BUDGET_FIELDS:
                if not math.isclose(
                    _float(coded, field), _float(baseline, field),
                    rel_tol=0.0, abs_tol=1e-6
                ):
                    raise ValueError("capacity-normalized storage budget mismatch")
        baseline_p99 = _float(baseline, "p99_us")
        coded_p99 = _float(coded, "p99_us")
        if baseline_p99 <= 0.0:
            raise ValueError("baseline P99 must be positive")
        delta = baseline_p99 - coded_p99
        absolute_deltas.append(delta)
        percentage_gains.append(100.0 * delta / baseline_p99)
        wins += coded_p99 < baseline_p99
    absolute_mean, absolute_ci95 = _mean_and_ci95(absolute_deltas)
    percentage_mean, percentage_ci95 = _mean_and_ci95(percentage_gains)
    return (
        absolute_mean,
        absolute_ci95,
        percentage_mean,
        percentage_ci95,
        100.0 * wins / len(absolute_deltas),
    )


def analyze_rows(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Phase 7 input contains no rows")

    required = {"seed", *CONFIG_FIELDS, *METRIC_FIELDS}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        _validate_budget(row)

    groups = defaultdict(list)
    raw_index = {}
    for row in rows:
        group_key = _config_key(row, CONFIG_FIELDS)
        groups[group_key].append(row)
        pair_key = (row["seed"], _config_key(row, PAIR_FIELDS), row["strategy"])
        if pair_key in raw_index:
            raise ValueError("duplicate seed within one experiment configuration")
        raw_index[pair_key] = row

    summary = []
    for key in sorted(groups):
        group_rows = groups[key]
        seeds = {row["seed"] for row in group_rows}
        if len(seeds) != len(group_rows):
            raise ValueError("duplicate seed within one experiment configuration")

        result = dict(zip(CONFIG_FIELDS, key))
        result["seed_count"] = len(seeds)
        for metric in METRIC_FIELDS:
            mean, ci95 = _mean_and_ci95([_float(row, metric) for row in group_rows])
            output_name = metric.removesuffix("_us").removesuffix("_pct")
            unit = "_us" if metric.endswith("_us") else "_pct"
            result[f"{output_name}_mean{unit}"] = mean
            result[f"{output_name}_ci95{unit}"] = ci95

        if result["strategy"] == "codedllm":
            for baseline in BASELINES:
                (
                    absolute_mean,
                    absolute_ci95,
                    percentage_mean,
                    percentage_ci95,
                    win_percent,
                ) = _paired_improvement(group_rows, raw_index, baseline)
                result[f"p99_delta_vs_{baseline}_mean_us"] = absolute_mean
                result[f"p99_delta_vs_{baseline}_ci95_us"] = absolute_ci95
                result[f"p99_gain_vs_{baseline}_mean_pct"] = percentage_mean
                result[f"p99_gain_vs_{baseline}_ci95_pct"] = percentage_ci95
                result[f"p99_win_vs_{baseline}_seed_pct"] = win_percent
        summary.append(result)
    return summary


METRIC_OUTPUT_FIELDS = tuple(
    field
    for metric in METRIC_FIELDS
    for field in (
        f"{metric.removesuffix('_us').removesuffix('_pct')}_mean"
        + ("_us" if metric.endswith("_us") else "_pct"),
        f"{metric.removesuffix('_us').removesuffix('_pct')}_ci95"
        + ("_us" if metric.endswith("_us") else "_pct"),
    )
)
COMPARISON_OUTPUT_FIELDS = tuple(
    field
    for baseline in BASELINES
    for field in (
        f"p99_delta_vs_{baseline}_mean_us",
        f"p99_delta_vs_{baseline}_ci95_us",
        f"p99_gain_vs_{baseline}_mean_pct",
        f"p99_gain_vs_{baseline}_ci95_pct",
        f"p99_win_vs_{baseline}_seed_pct",
    )
)
OUTPUT_FIELDS = (*CONFIG_FIELDS, "seed_count", *METRIC_OUTPUT_FIELDS, *COMPARISON_OUTPUT_FIELDS)
CONCLUSION_FIELDS = (
    *SCENARIO_FIELDS,
    "seed_count",
    "is_primary_scenario",
    "wait_p99_delta_mean_us",
    "wait_p99_delta_ci95_us",
    "wait_p99_interval_supported",
    "capacity_replication_p99_delta_mean_us",
    "capacity_replication_p99_delta_ci95_us",
    "capacity_replication_p99_interval_supported",
    "full_replication_p99_delta_mean_us",
    "full_replication_p99_delta_ci95_us",
    "full_replication_p99_interval_supported",
    "conclusion_scope",
)


def _interval_supported(row, baseline):
    return (
        row[f"p99_delta_vs_{baseline}_mean_us"]
        - row[f"p99_delta_vs_{baseline}_ci95_us"]
        > 0.0
    )


def build_conclusions(summary):
    conclusions = []
    for row in summary:
        if row["strategy"] != "codedllm":
            continue
        result = {field: row[field] for field in SCENARIO_FIELDS}
        result["seed_count"] = row["seed_count"]
        result["is_primary_scenario"] = row["scenario"] == "primary"
        for prefix, baseline in (
            ("wait", "wait"),
            ("capacity_replication", "replication_capacity_normalized"),
            ("full_replication", "replication_full"),
        ):
            result[f"{prefix}_p99_delta_mean_us"] = row[
                f"p99_delta_vs_{baseline}_mean_us"
            ]
            result[f"{prefix}_p99_delta_ci95_us"] = row[
                f"p99_delta_vs_{baseline}_ci95_us"
            ]
            result[f"{prefix}_p99_interval_supported"] = _interval_supported(
                row, baseline
            )
        result["conclusion_scope"] = (
            "primary_interval_supported"
            if result["is_primary_scenario"]
            and result["wait_p99_interval_supported"]
            else "exploratory" if not result["is_primary_scenario"] else "not_supported"
        )
        conclusions.append(result)
    return conclusions


def _format_output(row, fields):
    return {
        field: f"{row[field]:.6f}" if isinstance(row.get(field), float) else row.get(field, "")
        for field in fields
    }


def write_csv(destination, fields, rows):
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()
    writer.writerows(_format_output(row, fields) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="raw multi-seed Phase 7 CSV")
    parser.add_argument("-o", "--output", help="summary CSV; defaults to stdout")
    parser.add_argument("--conclusions", help="paired P99 conclusion CSV")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as source:
        summary = analyze_rows(csv.DictReader(source))

    destination = (
        open(args.output, "w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    try:
        write_csv(destination, OUTPUT_FIELDS, summary)
    finally:
        if args.output:
            destination.close()

    if args.conclusions:
        with open(args.conclusions, "w", newline="", encoding="utf-8") as destination:
            write_csv(destination, CONCLUSION_FIELDS, build_conclusions(summary))


if __name__ == "__main__":
    main()
