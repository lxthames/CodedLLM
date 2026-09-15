#!/usr/bin/env python3
"""Run a fixed, randomized 30-block GPU latency experiment; retain raw JSON.

Each path/degree is a fresh process in each block. Intervals quantify between-
invocation variation on this machine, not between-machine uncertainty.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

PATHS = ('EndToEndLatency', 'StagedReadyToResultLatency', 'KernelOnlyLatency')
DEGREES = (2, 3, 5, 8)
BLOCKS = 30
T29 = 2.045229642132703


def interval(values):
    if len(values) != BLOCKS or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected 30 finite block observations')
    mean = statistics.mean(values)
    half = T29 * statistics.stdev(values) / math.sqrt(BLOCKS)
    return dict(mean=mean, ci95_half_width=half, n=BLOCKS,
                minimum=min(values), maximum=max(values))


def extract(document, path, degree):
    rows = document['benchmarks']
    if len(rows) != 1 or rows[0].get('error_occurred'):
        raise ValueError('Expected one successful benchmark row')
    row = rows[0]
    if not row['name'].startswith(f'SparseDecoderBenchmark/{path}/{degree}/16777216'):
        raise ValueError('Unexpected benchmark configuration')
    expected = 100 if path == 'KernelOnlyLatency' else 20
    if row['iterations'] != expected:
        raise ValueError(f'Expected {expected} iterations, got {row["iterations"]}')
    scale = {'ns': 1e-6, 'us': 1e-3, 'ms': 1, 's': 1000}[row['time_unit']]
    value = row['real_time'] * scale
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid elapsed time')
    return value


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    observations = {}
    for item in manifest['schedule']:
        key = (item['block'], item['degree'], item['path'])
        if key in observations:
            raise ValueError('Duplicate observation')
        observations[key] = extract(json.loads((directory / item['file']).read_text()),
                                    item['path'], item['degree'])
    if len(observations) != BLOCKS * len(DEGREES) * len(PATHS):
        raise ValueError('Incomplete experiment')
    result = {'units': 'ms except improvement_pct', 'degrees': {}}
    for degree in DEGREES:
        entry = {path: interval([observations[b, degree, path] for b in range(BLOCKS)])
                 for path in PATHS}
        entry['paired_saving_ms'] = interval([
            observations[b, degree, PATHS[0]] - observations[b, degree, PATHS[1]]
            for b in range(BLOCKS)])
        entry['improvement_pct'] = interval([
            100 * (1 - observations[b, degree, PATHS[1]] / observations[b, degree, PATHS[0]])
            for b in range(BLOCKS)])
        result['degrees'][str(degree)] = entry
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def capture(command):
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    return {'command': command, 'returncode': proc.returncode, 'output': proc.stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build-ninja-container/bench_decoder')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize(args.output), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path(args.binary).resolve()
    rng = random.Random(20260915)
    schedule = []
    for block in range(BLOCKS):
        cases = [(d, p) for d in DEGREES for p in PATHS]
        rng.shuffle(cases)
        for degree, path in cases:
            schedule.append(dict(block=block, degree=degree, path=path,
                                 file=f'block-{block:02d}-{degree}-{path}.json'))
    manifest = dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    blocks=BLOCKS, order_seed=20260915, schedule=schedule,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path('benchmarks/bench_decoder.cpp').read_bytes()).hexdigest(),
                    transfer_iterations=20, kernel_iterations=100,
                    exclusions='None; abort on failed or malformed invocation',
                    interval='Two-sided pointwise Student t, df=29; independent-process block means; no multiplicity correction',
                    improvement='Mean of block-paired 100*(1-staged/end_to_end); not ratio of grand means',
                    metadata=[capture(c) for c in [
                        ['git', 'rev-parse', 'HEAD'], ['git', 'status', '--short'],
                        ['nvcc', '--version'], ['nvidia-smi'], ['lscpu'],
                        ['uname', '-a']]])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for index, item in enumerate(schedule):
        if index % 12 == 0:
            (args.output / f'gpu-before-block-{item["block"]:02d}.json').write_text(
                json.dumps(capture(['nvidia-smi']), indent=2) + '\n')
        command = [str(binary),
                   f'--benchmark_filter=^SparseDecoderBenchmark/{item["path"]}/{item["degree"]}/16777216(/|$)',
                   '--benchmark_min_time=20x', '--benchmark_repetitions=1',
                   '--benchmark_format=json']
        with (args.output / item['file']).open('w') as out, \
                (args.output / (item['file'] + '.stderr')).open('w') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)
        value = extract(json.loads((args.output / item['file']).read_text()),
                        item['path'], item['degree'])
        print(f'{index+1}/{len(schedule)} block={item["block"]} degree={item["degree"]} '
              f'{item["path"]}: {value:.6f} ms', flush=True)
    summarize(args.output)
    (args.output / 'completed.json').write_text(json.dumps({
        'end_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'observations': len(schedule)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
