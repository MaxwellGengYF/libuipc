#!/usr/bin/env python3
"""Drive the CUDA<->Luisa parity harness end to end (S0 / T9.3a).

One invocation:

  1. optionally runs the Tier-1 golden generator (uipc_parity_reference);
  2. runs `uipc_parity_runner` once per (scene, backend) -- ALWAYS in a separate
     process, because a second Engine in one process breaks the LC stream
     (.kimix_cache/plans/03-parity-harness.md §1). The child's stdout+stderr is kept
     as <leg>/run.log, and the backend module's md5 is recorded in run.json;
  3. runs `scripts/parity_compare.py` on the pair;
  4. writes <out>/run.json and <out>/summary.md.

Exit codes (research §3 D7):
    0  every leg compared and in tolerance
    1  a violation, a simulation error, or a non-finite state
    3  everything that could be compared was skipped (the reference backend is not
       built here -- this is a SKIP, never a pass)
    --strict turns 3 into 1 (CI uses --strict only in a job that builds both backends).

Usage:
    python scripts/parity_run.py --scene=harness_smoke --frames=3
    python scripts/parity_run.py --scene=gravity_abd --frames=3 --strict
    python scripts/parity_run.py --compare-only --ref A/luisa --test B/luisa
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMPARE = os.path.join(HERE, 'parity_compare.py')

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_USAGE = 2
EXIT_SKIPPED = 3

# Windows crash codes a child can return when a UIPC_ASSERT aborts inside do_build /
# do_init (it cannot be caught: the process dies). Python reports the raw NTSTATUS as
# a signed 32-bit int; bash/CI may report the unsigned form or 127.
CRASH_CODES = {
    3221226505,    # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN (__fastfail / assert)
    -1073740791,   # the same value as a signed int
    3221225477,    # 0xC0000005 access violation
    -1073741819,
    3221225786,    # 0xC000013A Ctrl-C
    -1073741510,
    3221225781,    # 0xC0000135 DLL not found
    -1073741515,
    127,
}

LEG_OK = 'ok'
LEG_SKIP = 'skip'
LEG_USAGE = 'usage'
LEG_SIM_ERROR = 'simulation_error'
LEG_NON_FINITE = 'non_finite'


def find_runner(explicit):
    if explicit:
        return os.path.abspath(explicit)
    patterns = [
        os.path.join(ROOT, 'build', 'windows', 'x64', 'release', 'uipc_parity_runner.exe'),
        os.path.join(ROOT, 'build', 'windows', 'x64', 'debug', 'uipc_parity_runner.exe'),
        os.path.join(ROOT, 'build', 'Release', 'bin', 'uipc_parity_runner.exe'),
        os.path.join(ROOT, 'build', 'RelWithDebInfo', 'bin', 'uipc_parity_runner.exe'),
        os.path.join(ROOT, 'build', 'Release', 'bin', 'uipc_parity_runner'),
        os.path.join(ROOT, 'build', 'RelWithDebInfo', 'bin', 'uipc_parity_runner'),
        os.path.join(ROOT, 'build', 'linux', 'x86_64', 'release', 'uipc_parity_runner'),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                return path
    return None


def classify(returncode):
    if returncode == 0:
        return LEG_OK
    if returncode == 3:
        return LEG_SKIP
    if returncode == 2:
        return LEG_USAGE
    if returncode == 5:
        return LEG_NON_FINITE
    if returncode in CRASH_CODES:
        return LEG_SIM_ERROR
    return LEG_SIM_ERROR


def file_digests(path):
    if not path or not os.path.isfile(path):
        return {'path': path, 'state': 'absent'}
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            md5.update(chunk)
            sha256.update(chunk)
    stat = os.stat(path)
    return {'path': path, 'state': 'present', 'size': stat.st_size,
            'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def module_path(runner, backend):
    base = os.path.dirname(runner)
    for name in ('uipc_backend_%s.dll' % backend, 'libuipc_backend_%s.so' % backend,
                 'uipc_backend_%s.so' % backend):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    return os.path.join(base, 'uipc_backend_%s.dll' % backend)


def tail(path, lines=20):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.readlines()
    except OSError:
        return []
    return [line.rstrip('\n') for line in content[-lines:]]


def run_leg(runner, scene, backend, frames, warmup, dt, out_root, timeout,
            dump_linear_system):
    leg_dir = os.path.join(out_root, scene, backend)
    os.makedirs(leg_dir, exist_ok=True)
    log_path = os.path.join(leg_dir, 'run.log')
    command = [runner, '--scene=%s' % scene, '--backend=%s' % backend,
               '--frames=%d' % frames, '--warmup=%d' % warmup,
               '--dt=%r' % dt, '--out=%s' % os.path.join(out_root, scene)]
    if dump_linear_system:
        command.append('--dump-linear-system')

    started = time.time()
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        log.write('# %s\n' % ' '.join(command))
        log.flush()
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            log.write('# TIMEOUT after %ss\n' % timeout)
            returncode = -1
        except OSError as error:
            log.write('# could not start the runner: %s\n' % error)
            returncode = -2
    duration = time.time() - started

    state = classify(returncode) if returncode >= 0 else LEG_SIM_ERROR
    leg = {
        'scene': scene,
        'backend': backend,
        'command': command,
        'returncode': returncode,
        'state': state,
        'duration_s': round(duration, 3),
        'log': log_path,
        'module': file_digests(module_path(runner, backend)),
    }
    if state != LEG_OK:
        leg['log_tail'] = tail(log_path)
    return leg


def run_reference(exe, timeout):
    started = time.time()
    try:
        completed = subprocess.run([exe], cwd=ROOT, capture_output=True, text=True,
                                   timeout=timeout)
        return {'command': [exe], 'returncode': completed.returncode,
                'duration_s': round(time.time() - started, 3),
                'stdout': completed.stdout.strip().splitlines()[-10:],
                'stderr': completed.stderr.strip().splitlines()[-10:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': [exe], 'returncode': -1, 'error': str(error)}


def compare(scene, out_root, extra_args=()):
    ref_dir = os.path.join(out_root, scene, 'cuda')
    test_dir = os.path.join(out_root, scene, 'luisa')
    command = [sys.executable, COMPARE, '--ref', ref_dir, '--test', test_dir,
               '--out', os.path.join(out_root, scene)] + list(extra_args)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        'scene': scene,
        'command': command,
        'returncode': completed.returncode,
        'stdout': completed.stdout.strip().splitlines(),
        'stderr': completed.stderr.strip().splitlines(),
    }


def write_summary(out_root, report):
    lines = []
    lines.append('# Parity run summary')
    lines.append('')
    lines.append('| Field | Value |')
    lines.append('|---|---|')
    lines.append('| runner | `%s` |' % report['runner'])
    lines.append('| scenes | %s |' % ', '.join(report['scenes']))
    lines.append('| backends | %s |' % ', '.join(report['backends']))
    lines.append('| frames | %d |' % report['frames'])
    lines.append('| verdict | **%s** |' % report['verdict'])
    lines.append('| exit code | %d |' % report['exit_code'])
    lines.append('')
    lines.append('## Legs')
    lines.append('')
    lines.append('| scene | backend | state | rc | seconds | module md5 |')
    lines.append('|---|---|---|---|---|---|')
    for leg in report['legs']:
        lines.append('| %s | %s | %s | %s | %s | `%s` |'
                     % (leg['scene'], leg['backend'], leg['state'], leg['returncode'],
                        leg['duration_s'], leg['module'].get('md5', '-')))
    lines.append('')
    if report['comparisons']:
        lines.append('## Comparisons')
        lines.append('')
        lines.append('| scene | rc | verdict |')
        lines.append('|---|---|---|')
        for entry in report['comparisons']:
            verdict = 'skipped' if entry['returncode'] == EXIT_SKIPPED else (
                'identical / in tolerance' if entry['returncode'] == EXIT_OK
                else 'VIOLATION')
            lines.append('| %s | %d | %s |' % (entry['scene'], entry['returncode'],
                                               verdict))
        lines.append('')
        for entry in report['comparisons']:
            if entry['returncode'] not in (EXIT_OK,):
                lines.append('### %s (rc %d)' % (entry['scene'], entry['returncode']))
                lines.append('')
                lines.append('```')
                lines.extend(entry['stdout'][:20])
                lines.extend(entry['stderr'][:20])
                lines.append('```')
                lines.append('')
    if report.get('reference'):
        lines.append('## Golden generator')
        lines.append('')
        lines.append('- `%s` -> rc %s'
                     % (' '.join(report['reference']['command']),
                        report['reference']['returncode']))
        lines.append('')
    failing = [leg for leg in report['legs'] if leg['state'] not in (LEG_OK, LEG_SKIP)]
    if failing:
        lines.append('## Leg failures (last log lines)')
        lines.append('')
        for leg in failing:
            lines.append('### %s / %s (rc %s)' % (leg['scene'], leg['backend'],
                                                 leg['returncode']))
            lines.append('')
            lines.append('```')
            lines.extend(leg.get('log_tail', [])[-12:])
            lines.append('```')
            lines.append('')
    path = os.path.join(out_root, 'summary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the CUDA<->Luisa parity harness (two processes, file based).')
    parser.add_argument('--runner', help='path to uipc_parity_runner(.exe)')
    parser.add_argument('--scene', action='append', default=[],
                        help='scene name (repeatable or comma separated)')
    parser.add_argument('--scenes', help='comma separated scene list')
    parser.add_argument('--backends', default='cuda,luisa',
                        help='comma separated backend list (default cuda,luisa)')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out', help='artifact root (default output/parity/<date>)')
    parser.add_argument('--timeout', type=int, default=900,
                        help='per-leg timeout in seconds (never below 300: the first '
                             'frame pays the full LC JIT, research §6 R-4)')
    parser.add_argument('--reference-exe', help='Tier-1 golden generator to run first')
    parser.add_argument('--skip-golden', action='store_true',
                        help='do not run the golden generator even if it is found')
    parser.add_argument('--dump-linear-system', action='store_true')
    parser.add_argument('--shard', help='i/n: run only shard i of the leg list')
    parser.add_argument('--strict', action='store_true',
                        help='treat "everything skipped" (rc 3) as a failure (rc 1)')
    parser.add_argument('--compare-only', action='store_true',
                        help='do not run any leg; only compare --ref with --test')
    parser.add_argument('--ref', help='--compare-only: reference artifact directory')
    parser.add_argument('--test', help='--compare-only: test artifact directory')
    parser.add_argument('--list', action='store_true',
                        help='print the runner scene registry and exit')
    args = parser.parse_args(argv)

    if args.timeout < 300:
        print('warning: --timeout below 300 s; the first frame JITs every LC pipeline',
              file=sys.stderr)

    runner = find_runner(args.runner)

    if args.list:
        if not runner:
            print('uipc_parity_runner not found; pass --runner', file=sys.stderr)
            return EXIT_USAGE
        completed = subprocess.run([runner, '--list'], cwd=ROOT, capture_output=True,
                                   text=True)
        print(completed.stdout, end='')
        return completed.returncode

    if args.compare_only:
        if not args.ref or not args.test:
            print('--compare-only needs --ref and --test', file=sys.stderr)
            return EXIT_USAGE
        command = [sys.executable, COMPARE, '--ref', args.ref, '--test', args.test]
        if args.out:
            command += ['--out', args.out]
        return subprocess.run(command, cwd=ROOT).returncode

    if not runner:
        print('uipc_parity_runner not found: build the parity_runner target or pass '
              '--runner', file=sys.stderr)
        return EXIT_USAGE

    scenes = []
    for chunk in args.scene + ([args.scenes] if args.scenes else []):
        scenes.extend(name for name in chunk.split(',') if name)
    if not scenes:
        completed = subprocess.run([runner, '--list'], cwd=ROOT, capture_output=True,
                                   text=True)
        if completed.returncode != 0:
            print('the runner cannot list its scenes:\n%s' % completed.stderr,
                  file=sys.stderr)
            return EXIT_USAGE
        scenes = [line.split()[0] for line in completed.stdout.splitlines() if line.split()]

    backends = [name for name in args.backends.split(',') if name]
    out_root = args.out or os.path.join(ROOT, 'output', 'parity',
                                        time.strftime('%Y-%m-%d'))

    legs_to_run = [(scene, backend) for scene in scenes for backend in backends]
    if args.shard:
        try:
            index, count = (int(part) for part in args.shard.split('/'))
        except ValueError:
            print('--shard expects i/n', file=sys.stderr)
            return EXIT_USAGE
        if count < 1 or not 0 <= index < count:
            print('--shard %s is out of range' % args.shard, file=sys.stderr)
            return EXIT_USAGE
        legs_to_run = [leg for position, leg in enumerate(legs_to_run)
                       if position % count == index]

    report = {
        'runner': runner,
        'scenes': scenes,
        'backends': backends,
        'frames': args.frames,
        'out': out_root,
        'legs': [],
        'comparisons': [],
    }

    if not args.skip_golden:
        reference = args.reference_exe
        if not reference:
            candidate = os.path.join(os.path.dirname(runner), 'uipc_parity_reference.exe')
            if os.path.isfile(candidate):
                reference = candidate
        if reference:
            report['reference'] = run_reference(reference, args.timeout)

    for scene, backend in legs_to_run:
        leg = run_leg(runner, scene, backend, args.frames, args.warmup, args.dt,
                      out_root, args.timeout, args.dump_linear_system)
        print('%-14s %-6s %-16s rc=%-4s %.1fs'
              % (scene, backend, leg['state'], leg['returncode'], leg['duration_s']))
        report['legs'].append(leg)

    states = {(leg['scene'], leg['backend']): leg['state'] for leg in report['legs']}
    for scene in scenes:
        if args.shard:
            present = [backend for backend in backends if (scene, backend) in states]
            if len(present) != len(backends):
                continue
        entry = compare(scene, out_root)
        report['comparisons'].append(entry)
        print('%-14s compare rc=%d' % (scene, entry['returncode']))
        for line in entry['stdout'][-4:]:
            print('    %s' % line)

    any_violation = any(entry['returncode'] == EXIT_VIOLATION
                        for entry in report['comparisons'])
    any_error = any(leg['state'] in (LEG_USAGE, LEG_SIM_ERROR, LEG_NON_FINITE)
                    for leg in report['legs'])
    comparable = [entry for entry in report['comparisons']]
    all_skipped = bool(comparable) and all(entry['returncode'] == EXIT_SKIPPED
                                           for entry in comparable)

    if any_violation or any_error:
        exit_code = EXIT_VIOLATION
        report['verdict'] = 'violation' if any_violation else 'leg failure'
    elif all_skipped:
        exit_code = EXIT_VIOLATION if args.strict else EXIT_SKIPPED
        report['verdict'] = 'skipped (reference backend absent)'
    elif not comparable:
        exit_code = EXIT_SKIPPED
        report['verdict'] = 'nothing compared'
    else:
        exit_code = EXIT_OK
        report['verdict'] = 'identical'
    report['exit_code'] = exit_code

    os.makedirs(out_root, exist_ok=True)
    summary_path = write_summary(out_root, report)
    with open(os.path.join(out_root, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print('verdict: %s -> exit %d' % (report['verdict'], exit_code))
    print('report: %s' % summary_path)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
