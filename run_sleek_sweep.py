#!/usr/bin/env python3
"""Benchmark lossy SLEEK at ten L-inf caps from 0.01 through 10.
For each cap, measure an L-inf-only baseline and use its whole-file L2
error to set the search budgets. Verify errors after every round trip.
Use --l2-budget B for a fixed absolute budget instead.
"""
import argparse
from array import array
import csv
import filecmp
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

FIELDS = [
    'file', 'command', 'command_args', 'decompress_command', 'decompress_args',
    'original_bytes', 'compressed_bytes',
    'compression_ratio', 'exact_reconstruction', 'compression_time_seconds',
    'compression_throughput_MB_s', 'decompression_time_seconds',
    'l2_budget', 'absolute_l_inf_cap', 'selected_epsilon', 'search_trials',
    'decompression_throughput_MB_s', 'mode', 'epsilon', 'rmse', 'l_inf', 'l2',
    'budget_fraction', 'baseline_l2', 'baseline_cr', 'status', 'failure',
]
DEFAULT_FRACTIONS = [1.1, 0.9, 0.5, 0.1]
EPSILONS = [0.01 * 1000 ** (i / 9) for i in range(10)]
EPSILONS[0], EPSILONS[-1] = 0.01, 10.0


def timed_run(command, timeout):
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {shlex.join(command)}\n'
                           f'{result.stdout}\n{result.stderr}')
    return elapsed, result.stdout


def reconstruction_errors(original, restored, precision):
    """Measure native binary floats in bounded chunks using double arithmetic.

    hypot avoids overflow/underflow from directly summing squared differences.
    L2 is the Euclidean norm; RMSE is L2 / sqrt(number of values).
    """
    typecode = 'f' if precision == 'single' else 'd'
    width = array(typecode).itemsize
    count = 0
    l_inf = l2 = 0.0
    with open(original, 'rb') as src, open(restored, 'rb') as dst:
        while True:
            raw = src.read(65536 * width)
            reconstructed = dst.read(65536 * width)
            if len(raw) != len(reconstructed) or len(raw) % width:
                raise RuntimeError('Invalid or mismatched reconstructed float data')
            if not raw:
                break
            values, decoded = array(typecode), array(typecode)
            values.frombytes(raw)
            decoded.frombytes(reconstructed)
            errors = []
            for a, b in zip(values, decoded):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise RuntimeError('Cannot measure error for NaN or infinity')
                error = abs(a - b)
                if not math.isfinite(error):
                    raise RuntimeError('Reconstruction difference exceeds float64 range')
                errors.append(error)
            l_inf = max(l_inf, max(errors))
            l2 = math.hypot(l2, math.hypot(*errors))
            count += len(values)
    if not count or not math.isfinite(l2):
        raise RuntimeError('Empty input or L2 norm exceeds float64 range')
    return dict(rmse=l2 / math.sqrt(count), l_inf=l_inf, l2=l2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Binary float32 or float64 file')
    parser.add_argument('-o', '--output', type=Path, default=Path('sleek_results.csv'))
    parser.add_argument('--append', action='store_true',
                        help='Append to an existing results CSV (header must match)')
    parser.add_argument('--precision', choices=['single', 'double'], default='single')
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--bin-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Directory containing standalone SLEEK executables')
    parser.add_argument('--timeout', type=float, default=300,
                        help='Maximum seconds per compressor/decompressor command (default: 300)')
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument('--budget-fractions', nargs='+', type=float, default=DEFAULT_FRACTIONS,
                         help='Multipliers of baseline L2 (default: 1.1 0.9 0.5 0.1)')
    budgets.add_argument('--l2-budget', type=float,
                        help='Search each L-inf sweep point with this absolute whole-file L2 budget')
    args = parser.parse_args()
    if args.l2_budget is not None and (not math.isfinite(args.l2_budget) or args.l2_budget < 0):
        parser.error('--l2-budget must be finite and nonnegative')
    if any(not math.isfinite(f) or f < 0 for f in args.budget_fractions):
        parser.error('Budget fractions must be finite and nonnegative')
    src = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not src.is_file():
        parser.error(f'Input file does not exist: {src}')
    original_bytes = src.stat().st_size
    width = 4 if args.precision == 'single' else 8
    if original_bytes == 0 or original_bytes % width:
        parser.error(f'Input must be nonempty and a multiple of {width} bytes')
    if args.timeout <= 0 or not args.timeout < float('inf'):
        parser.error('--timeout must be positive and finite')
    if output == src:
        parser.error('Output CSV must differ from the input file')
    suffix = '_cpu' if args.backend == 'cpu' else ''
    binaries = {}
    for operation in ['compressor', 'decompressor']:
        path = args.bin_dir.expanduser().resolve() / f'sleek_{args.precision}_{operation}_lossy{suffix}'
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f'Missing executable: {path}. Compile standalone SLEEK binaries first.')
        if output == path:
            parser.error('Output CSV must differ from codec executables')
        binaries[operation] = str(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ARTIFACT builds require the original input for lossy decompression.
    usage = subprocess.run([binaries['decompressor'], '--help'],
                           capture_output=True, text=True, timeout=args.timeout)
    needs_original = 'original_file_name' in usage.stdout + usage.stderr
    has_header = args.append and output.exists() and output.stat().st_size > 0
    if has_header:
        with output.open(newline='') as existing:
            if next(csv.reader(existing), None) != FIELDS:
                parser.error('Existing output CSV has an incompatible header')
    # Default exclusive creation protects reports; batch runs explicitly append.
    with output.open('a' if args.append else 'x', newline='') as report, tempfile.TemporaryDirectory(prefix='sleek-sweep-') as work:
        writer = csv.DictWriter(report, fieldnames=FIELDS)
        if not has_header:
            writer.writeheader()
        report.flush()
        compressed = Path(work) / 'compressed.bin'
        restored = Path(work) / 'restored.bin'
        failures = 0
        row_count = 0

        def run_case(epsilon, cap, budget=None, fraction=None, baseline=None):
            nonlocal failures, row_count
            mode = 'lossy_l_inf_only' if budget is None else 'lossy_dual_bound_search'
            comp = [binaries['compressor'], str(src), str(compressed), str(epsilon)]
            decomp = [binaries['decompressor'], str(compressed), str(restored), str(epsilon)]
            if needs_original:
                decomp.append(str(src))
            if budget is not None:
                comp.extend(['--l2-budget', str(budget)])
            row = dict(file=src.name, command=Path(comp[0]).name,
                       command_args=shlex.join([Path(arg).name for arg in comp[1:]]),
                       decompress_command=Path(decomp[0]).name,
                       decompress_args=shlex.join([Path(arg).name for arg in decomp[1:]]),
                       mode=mode, epsilon=epsilon, absolute_l_inf_cap=cap,
                       original_bytes=original_bytes, l2_budget=budget, budget_fraction=fraction)
            if baseline is not None:
                row.update(baseline_l2=baseline['l2'], baseline_cr=baseline['compression_ratio'])
            try:
                if budget is not None and not math.isfinite(budget):
                    raise ValueError('Computed L2 budget exceeds float64 range')
                compressed.unlink(missing_ok=True)
                restored.unlink(missing_ok=True)
                elapsed, log = timed_run(comp, args.timeout)
                size = compressed.stat().st_size
                if size == 0:
                    raise RuntimeError('Empty compressed file')
                row.update(compressed_bytes=size, compression_ratio=original_bytes / size,
                           compression_time_seconds=elapsed,
                           compression_throughput_MB_s=original_bytes / 1e6 / elapsed)
                if budget is not None:
                    stats = dict(line.split(': ', 1) for line in log.splitlines() if ': ' in line)
                    row.update(selected_epsilon=float(stats['selected epsilon']),
                               search_trials=int(stats['search trials']))
                elapsed, _ = timed_run(decomp, args.timeout)
                row.update(reconstruction_errors(src, restored, args.precision))
                row.update(decompression_time_seconds=elapsed,
                           decompression_throughput_MB_s=original_bytes / 1e6 / elapsed,
                           exact_reconstruction=filecmp.cmp(src, restored, shallow=False))
                filecmp.clear_cache()
                if row['l_inf'] > cap or (budget is not None and row['l2'] > budget):
                    raise RuntimeError('Measured reconstruction error exceeds requested bound')
                row['status'] = 'pass'
            except (RuntimeError, ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
                failures += 1
                row.update(status='fail', failure=str(exc).splitlines()[-1][:180])
                print(f'{src.name}, epsilon={epsilon}, budget={budget}: {exc}', file=sys.stderr)
            writer.writerow(row)
            report.flush()
            row_count += 1
            return row

        input_range = None
        if needs_original:
            # ARTIFACT codecs interpret epsilon relative to the input range.
            minimum, maximum = math.inf, -math.inf
            with src.open('rb') as data:
                while raw := data.read(65536 * width):
                    values = array('f' if args.precision == 'single' else 'd')
                    values.frombytes(raw)
                    minimum = min(minimum, min(values))
                    maximum = max(maximum, max(values))
            input_range = maximum - minimum
        for epsilon in EPSILONS:
            cap = epsilon if input_range is None else epsilon * input_range
            baseline = run_case(epsilon, cap)
            if args.l2_budget is not None:
                run_case(epsilon, cap, args.l2_budget,
                         baseline=baseline if 'l2' in baseline else None)
            elif 'l2' in baseline:
                e0 = baseline['l2']
                if e0 == 0:
                    run_case(epsilon, cap, 0.0, baseline=baseline)
                else:
                    for fraction in dict.fromkeys(args.budget_fractions):
                        run_case(epsilon, cap, fraction * e0, fraction, baseline)
    print(f'{src.name}: {row_count} attempted runs, {failures} failed. Results: {output}')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
