#!/usr/bin/env python3
"""Compare two parity artifacts (S0 / T9.3a, .kimix_cache/plans/03-parity-harness.md §5).

The comparison is file based and class aware:

  * the per-frame checksum recorded by the runner is recomputed from the raw arrays,
    so a broken writer cannot pass as "identical";
  * integer/boolean/string scalars must be exactly equal, floats use the recorded
    tolerance band;
  * every probe array reports max_abs / max_rel / count-over / worst index, so a
    failure names the probe *and* the element index;
  * Matrix-Market dumps (when `--dump-linear-system` produced them) are compared by
    coalesced (i, j) sparsity first and by value second -- never by re-sorting.

Usage:
    python scripts/parity_compare.py --ref DIR --test DIR [--out DIR]
    python scripts/parity_compare.py --check-tolerances
    python scripts/parity_compare.py --selftest

Exit codes: 0 in tolerance, 1 violation, 2 usage, 3 skipped (a SKIP.json on either
side: the reference backend is not built here).

The script is stdlib only on purpose: this machine has no numpy/scipy and the goal is
that `python scripts/parity_compare.py` works in any CI job (research §6 R-5).
"""
import argparse
import array
import csv
import json
import os
import struct
import sys
import tempfile

SCHEMA = 'uipc-parity/1'
TOLERANCE_SCHEMA = 'uipc-parity-tolerances/1'

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_USAGE = 2
EXIT_SKIPPED = 3

DEFAULT_TOLERANCES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'apps', 'tests', 'parity', 'tolerances.json')

# Built-in defaults; apps/tests/parity/tolerances.json must agree with them and every
# *override* in that file must carry a `_why` entry (--check-tolerances).
BUILTIN_DEFAULTS = {
    'abs': 0.0,            # bit-exact by default: the strongest possible gate
    'rel': 0.0,
    'eps_scale': 1.0,
    'scalars': {
        'float_abs': 0.0,
        'float_rel': 0.0,
        'ignore': [],
    },
}


class Comparison:
    """Accumulates violations and the per-probe rows of the report."""

    def __init__(self):
        self.rows = []
        self.violations = []
        self.notes = []

    def note(self, text):
        self.notes.append(text)

    def row(self, **fields):
        self.rows.append(fields)

    def violate(self, text):
        self.violations.append(text)

    @property
    def ok(self):
        return not self.violations


# ---------------------------------------------------------------------------
#  small IO helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_f64(path):
    """Raw little-endian float64 (the artifact format, 03-parity-harness.md §3.3)."""
    if sys.byteorder != 'little':  # pragma: no cover - both legs are x86/ARM-LE
        raise SystemExit('parity_compare.py assumes a little-endian host')
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) % 8 != 0:
        raise ValueError('%s is %d bytes, not a multiple of 8' % (path, len(data)))
    values = array.array('d')
    values.frombytes(data)
    return values


# ---------------------------------------------------------------------------
#  tolerances
# ---------------------------------------------------------------------------
def load_tolerances(path):
    tol = {
        'defaults': json.loads(json.dumps(BUILTIN_DEFAULTS)),
        'probes': {},
        'scenes': {},
        '_why': {},
    }
    if path and os.path.isfile(path):
        data = load_json(path)
        schema = data.get('schema')
        if schema != TOLERANCE_SCHEMA:
            raise SystemExit('%s: unexpected schema %r (want %r)'
                             % (path, schema, TOLERANCE_SCHEMA))
        defaults = dict(tol['defaults'])
        defaults.update(data.get('defaults', {}))
        defaults['scalars'] = dict(BUILTIN_DEFAULTS['scalars'])
        defaults['scalars'].update(data.get('defaults', {}).get('scalars', {}))
        tol['defaults'] = defaults
        tol['probes'] = data.get('probes', {})
        tol['scenes'] = data.get('scenes', {})
        tol['_why'] = data.get('_why', {})
        tol['path'] = path
    return tol


def _numeric_leaves(node, prefix=''):
    """Yield (dotted path, value) for every numeric leaf of a tolerance subtree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == '_why':
                continue
            yield from _numeric_leaves(value, prefix + ('.' if prefix else '') + key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix, node


def check_tolerances(tol):
    """Every number outside `defaults` needs a `_why` sibling entry (research §3 D6)."""
    problems = []
    why = tol.get('_why', {})
    for section in ('probes', 'scenes'):
        for path, value in _numeric_leaves(tol.get(section, {}), section):
            if path not in why:
                problems.append('%s = %r has no _why entry' % (path, value))
    return problems


def effective_tolerance(tol, scene, probe):
    """Scene override > probe override > defaults, shallow-merged."""
    result = dict(tol['defaults'])
    result['scalars'] = dict(tol['defaults'].get('scalars', {}))
    for section in ('probes', 'scenes'):
        node = tol.get(section, {})
        if section == 'probes':
            override = node.get(probe, {})
        else:
            override = node.get(scene, {}).get('probes', {}).get(probe, {})
        for key, value in override.items():
            if key == 'scalars' and isinstance(value, dict):
                result['scalars'].update(value)
            else:
                result[key] = value
    return result


def scene_scalars_tolerance(tol, scene):
    """Scalar-level tolerance: defaults, then the scene's `scalars` block."""
    result = dict(tol['defaults'].get('scalars', {}))
    override = tol.get('scenes', {}).get(scene, {}).get('scalars', {})
    result.update(override)
    return result


# ---------------------------------------------------------------------------
#  array comparison (Class A/B/C)
# ---------------------------------------------------------------------------
def compare_arrays(ref, test, tolerance):
    """Class-aware element comparison of two equally shaped arrays."""
    abs_tol = float(tolerance.get('abs', 0.0))
    rel_tol = float(tolerance.get('rel', 0.0))
    eps_scale = float(tolerance.get('eps_scale', 1.0))

    n = min(len(ref), len(test))
    max_abs = 0.0
    max_rel = 0.0
    count_over = 0
    worst_index = -1
    non_finite = 0
    first = None

    for i in range(n):
        a = ref[i]
        b = test[i]
        if a != a or b != b:  # NaN on either side
            if a != a and b != b:
                continue
            non_finite += 1
            delta = float('inf')
        elif a in (float('inf'), float('-inf')) or b in (float('inf'), float('-inf')):
            if a == b:
                continue
            non_finite += 1
            delta = float('inf')
        else:
            delta = abs(a - b)
        scale = max(abs(b), eps_scale)
        rel = delta / scale if scale > 0.0 else delta
        if delta > max_abs or worst_index < 0:
            max_abs = delta
            worst_index = i
        max_rel = max(max_rel, rel)
        if delta > abs_tol + rel_tol * scale:
            count_over += 1
            if first is None:
                first = (i, b, a, delta)

    return {
        'count': n,
        'max_abs': max_abs,
        'max_rel': max_rel,
        'count_over': count_over,
        'worst_index': worst_index,
        'non_finite': non_finite,
        'first': first,
        'shape_mismatch': len(ref) != len(test),
    }


# ---------------------------------------------------------------------------
#  Matrix Market (no scipy: the format is fixed, see
#  src/backends/cuda/utils/matrix_market.h:97-108)
# ---------------------------------------------------------------------------
def parse_mtx(path):
    """Parse a coordinate-real Matrix Market file into a coalesced dict.

    Entries are keyed by 1-based (i, j) and duplicates are *summed* (coalesced), which
    is what both backends are required to do; the duplicate count per cell is kept so
    a sparsity difference in the duplicate structure is still visible.
    """
    entries = {}
    counts = {}
    header = None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            if header is None:
                header = stripped
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
            entries[(i, j)] = entries.get((i, j), 0.0) + v
            counts[(i, j)] = counts.get((i, j), 0) + 1
    return {'header': header, 'entries': entries, 'counts': counts}


def compare_mtx(name, ref_dir, test_dir, comparison, tolerance):
    ref_path = os.path.join(ref_dir, name)
    test_path = os.path.join(test_dir, name)
    ref_exists = os.path.isfile(ref_path)
    test_exists = os.path.isfile(test_path)
    if not ref_exists and not test_exists:
        comparison.note('%s: absent on both sides (the dump gate did not run)' % name)
        return
    if ref_exists != test_exists:
        comparison.violate('%s: present on %s only'
                           % (name, 'ref' if ref_exists else 'test'))
        return
    ref = parse_mtx(ref_path)
    test = parse_mtx(test_path)
    if ref['counts'] != test['counts']:
        missing = sorted(set(ref['counts']) - set(test['counts']))[:5]
        extra = sorted(set(test['counts']) - set(ref['counts']))[:5]
        comparison.violate('%s: sparsity mismatch (ref-only %r, test-only %r)'
                           % (name, missing, extra))
        return
    keys = sorted(ref['entries'])
    ref_values = [ref['entries'][k] for k in keys]
    test_values = [test['entries'][k] for k in keys]
    result = compare_arrays(ref_values, test_values, tolerance)
    comparison.row(frame=-1, probe=name, geometry=-1, count=len(keys),
                   max_abs=result['max_abs'], max_rel=result['max_rel'],
                   count_over=result['count_over'],
                   worst_index=result['worst_index'],
                   status='ok' if result['count_over'] == 0 else 'violation')
    if result['count_over']:
        idx = result['worst_index']
        comparison.violate('%s: %d/%d values out of tolerance, worst (i,j)=(%d,%d) '
                           'ref=%.17g test=%.17g delta=%.3e'
                           % (name, result['count_over'], len(keys), keys[idx][0],
                              keys[idx][1], ref_values[idx], test_values[idx],
                              result['max_abs']))


# ---------------------------------------------------------------------------
#  scalars
# ---------------------------------------------------------------------------
def flatten(node, prefix=''):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, prefix + ('.' if prefix else '') + key))
    else:
        out[prefix] = node
    return out


def _ignored(key, ignore):
    for pattern in ignore:
        if key == pattern or key.startswith(pattern + '.'):
            return True
    return False


def compare_scalars(ref_dir, test_dir, scene, tol, comparison):
    ref_path = os.path.join(ref_dir, 'scalars.jsonl')
    test_path = os.path.join(test_dir, 'scalars.jsonl')
    if not (os.path.isfile(ref_path) and os.path.isfile(test_path)):
        comparison.note('scalars: one side has no scalars.jsonl; structural gate skipped')
        return
    tolerance = scene_scalars_tolerance(tol, scene)
    ignore = list(tolerance.get('ignore', []))
    float_abs = float(tolerance.get('float_abs', 0.0))
    float_rel = float(tolerance.get('float_rel', 0.0))

    ref_frames = [json.loads(line) for line in open(ref_path, encoding='utf-8') if line.strip()]
    test_frames = [json.loads(line) for line in open(test_path, encoding='utf-8') if line.strip()]
    if len(ref_frames) != len(test_frames):
        comparison.violate('scalars: %d frames on ref, %d on test'
                           % (len(ref_frames), len(test_frames)))

    for index, (ref_frame, test_frame) in enumerate(zip(ref_frames, test_frames)):
        if ref_frame.get('frame') != test_frame.get('frame'):
            comparison.violate('scalars: frame index mismatch at line %d (%r vs %r)'
                               % (index, ref_frame.get('frame'), test_frame.get('frame')))
        ref_stats = flatten(ref_frame.get('stats', {}))
        test_stats = flatten(test_frame.get('stats', {}))
        keys = sorted(set(ref_stats) | set(test_stats))
        checked = 0
        for key in keys:
            if _ignored(key, ignore):
                continue
            if key not in ref_stats or key not in test_stats:
                comparison.violate('scalars[frame %d]: %s present on one side only'
                                   % (index, key))
                continue
            a = test_stats[key]
            b = ref_stats[key]
            if isinstance(b, bool) or isinstance(a, bool):
                if a is not b:
                    comparison.violate('scalars[frame %d]: %s %r != %r (bool)'
                                       % (index, key, a, b))
            elif isinstance(b, str) or isinstance(a, str):
                if a != b:
                    comparison.violate('scalars[frame %d]: %s %r != %r (string)'
                                       % (index, key, a, b))
            elif isinstance(b, int) and isinstance(a, int):
                if a != b:
                    comparison.violate('scalars[frame %d]: %s %d != %d (count)'
                                       % (index, key, a, b))
            else:
                delta = abs(float(a) - float(b))
                scale = max(abs(float(b)), 1.0)
                if delta > float_abs + float_rel * scale:
                    comparison.violate('scalars[frame %d]: %s %.17g != %.17g (float, '
                                       'delta %.3e)' % (index, key, a, b, delta))
            checked += 1
        comparison.row(frame=index, probe='scalars', geometry=-1, count=checked,
                       max_abs=0.0, max_rel=0.0, count_over=0, worst_index=-1,
                       status='ok' if comparison.ok else 'violation')


# ---------------------------------------------------------------------------
#  manifest cross-check + checksum verification
# ---------------------------------------------------------------------------
def frame_checksum(frame_arrays, directory):
    """The FNV-1a / 53-bit digest the runner records for one frame.

    Recomputed here from the raw arrays (never trusted from the manifest): a writer
    that truncates or reorders its own payload must not be able to pass as
    "identical". The hashed field order is the C++ `arrays_checksum()` order --
    probe name, geometry index, element count, then every element.
    """
    mask = (1 << 64) - 1
    prime = 1099511628211
    digest = 14695981039346656037
    for entry in frame_arrays:
        for byte in entry['probe'].encode('utf-8'):
            digest = ((digest ^ byte) * prime) & mask
        geom = int(entry.get('geometry', -1))
        for shift in range(0, 64, 8):
            digest = ((digest ^ ((geom >> shift) & 0xFF)) * prime) & mask
        values = read_f64(os.path.join(directory, entry['file']))
        count = len(values)
        for shift in range(0, 64, 8):
            digest = ((digest ^ ((count >> shift) & 0xFF)) * prime) & mask
        for value in values:
            bits = struct.unpack('<Q', struct.pack('<d', value))[0]
            for shift in range(0, 64, 8):
                digest = ((digest ^ ((bits >> shift) & 0xFF)) * prime) & mask
    return float(digest >> 11)


def verify_checksums(manifest, directory, comparison, label):
    """Recompute every recorded checksum from the raw arrays (FNV-1a, 53-bit)."""
    mismatched = set()
    for index, (frame_arrays, recorded) in enumerate(zip(manifest.get('arrays', []),
                                                         manifest.get('checksums', []))):
        recomputed = frame_checksum(frame_arrays, directory)
        if recomputed != recorded.get('value'):
            mismatched.add(index)
            comparison.violate('%s: frame %d checksum recorded %r but the raw arrays '
                               'give %r (writer or artifact corruption)'
                               % (label, index, recorded.get('value'), recomputed))
    return mismatched


def check_manifest_pair(ref, test, comparison):
    for key in ('scene', 'scene_hash', 'frames', 'backend'):
        if key in ref or key in test:
            if ref.get(key) != test.get(key):
                comparison.violate('manifest: %s differs (ref %r, test %r)'
                                   % (key, ref.get(key), test.get(key)))
    ref_arrays = ref.get('arrays', [])
    test_arrays = test.get('arrays', [])
    if len(ref_arrays) != len(test_arrays):
        comparison.violate('manifest: %d recorded frames on ref, %d on test'
                           % (len(ref_arrays), len(test_arrays)))
        return
    for index, (ref_frame, test_frame) in enumerate(zip(ref_arrays, test_arrays)):
        ref_keys = sorted((e['probe'], e.get('geometry', -1), e['components'], e['count'])
                          for e in ref_frame)
        test_keys = sorted((e['probe'], e.get('geometry', -1), e['components'], e['count'])
                           for e in test_frame)
        if ref_keys != test_keys:
            comparison.violate('manifest: frame %d probe set differs (ref %r, test %r)'
                               % (index, ref_keys, test_keys))


# ---------------------------------------------------------------------------
#  the driver
# ---------------------------------------------------------------------------
def compare_dirs(ref_dir, test_dir, tol, out_dir=None):
    comparison = Comparison()

    for side, directory in (('ref', ref_dir), ('test', test_dir)):
        skip_path = os.path.join(directory, 'SKIP.json')
        if os.path.isfile(skip_path):
            skip = load_json(skip_path)
            comparison.note('SKIP: %s %s' % (side, skip.get('reason', 'skipped')))
            comparison.row(frame=-1, probe='skip', geometry=-1, count=0, max_abs=0.0,
                           max_rel=0.0, count_over=0, worst_index=-1,
                           status='skipped')
            summary = {'status': 'skipped',
                       'scene': skip.get('scene', '?'),
                       'backend': skip.get('backend', '?'),
                       'frames': 0,
                       'identical_frames': 0,
                       'total_frames': 0,
                       'reason': skip.get('reason', 'skipped'),
                       'side': side,
                       'violations': [],
                       'notes': list(comparison.notes)}
            # A skipped comparison still writes a report: the CI summary links it, and
            # "skipped" must be visible in the artifact, not only in the exit code.
            if out_dir:
                write_reports(comparison, summary, out_dir)
            return comparison, summary

    ref_manifest_path = os.path.join(ref_dir, 'manifest.json')
    test_manifest_path = os.path.join(test_dir, 'manifest.json')
    for path in (ref_manifest_path, test_manifest_path):
        if not os.path.isfile(path):
            comparison.violate('missing manifest %s' % path)
            return comparison, {'status': 'error',
                                'violations': list(comparison.violations)}

    ref = load_json(ref_manifest_path)
    test = load_json(test_manifest_path)
    for name, manifest in (('ref', ref), ('test', test)):
        if manifest.get('schema') != SCHEMA:
            comparison.violate('%s manifest schema %r (want %r)'
                               % (name, manifest.get('schema'), SCHEMA))

    check_manifest_pair(ref, test, comparison)

    scene = ref.get('scene', '')
    ref_bad = verify_checksums(ref, ref_dir, comparison, 'ref')
    test_bad = verify_checksums(test, test_dir, comparison, 'test')
    unverified = ref_bad | test_bad

    identical_frames = 0
    ref_checksums = [c.get('value') for c in ref.get('checksums', [])]
    test_checksums = [c.get('value') for c in test.get('checksums', [])]
    for index, (a, b) in enumerate(zip(ref_checksums, test_checksums)):
        if a == b and index not in unverified:
            identical_frames += 1
        elif index in unverified:
            comparison.note('frame %d: the recorded checksum does not match its own '
                            'raw arrays, so the frame is not counted as identical'
                            % index)
        else:
            comparison.note('frame %d: checksum differs (ref %.17g, test %.17g) '
                            '-- falling through to the per-probe comparison'
                            % (index, a, b))

    for index, (ref_frame, test_frame) in enumerate(zip(ref.get('arrays', []),
                                                        test.get('arrays', []))):
        for ref_entry, test_entry in zip(ref_frame, test_frame):
            probe = ref_entry['probe']
            tolerance = effective_tolerance(tol, scene, probe)
            ref_values = read_f64(os.path.join(ref_dir, ref_entry['file']))
            test_values = read_f64(os.path.join(test_dir, test_entry['file']))
            result = compare_arrays(ref_values, test_values, tolerance)
            comparison.row(frame=index, probe=probe,
                           geometry=ref_entry.get('geometry', -1),
                           count=result['count'],
                           max_abs=result['max_abs'],
                           max_rel=result['max_rel'],
                           count_over=result['count_over'],
                           worst_index=result['worst_index'],
                           status='ok' if result['count_over'] == 0 else 'violation')
            if result['shape_mismatch']:
                comparison.violate('frame %d %s: element count differs (%d vs %d)'
                                   % (index, probe, len(ref_values), len(test_values)))
            if result['count_over']:
                worst = result['worst_index']
                comparison.violate(
                    'frame %d %s[%d] (geometry %d): ref=%.17g test=%.17g delta=%.3e '
                    '> abs=%r rel=%r  [%d/%d elements over]'
                    % (index, probe, worst, ref_entry.get('geometry', -1),
                       ref_values[worst], test_values[worst], result['max_abs'],
                       tolerance.get('abs'), tolerance.get('rel'),
                       result['count_over'], result['count']))
            elif result['non_finite']:
                comparison.violate('frame %d %s: %d non-finite mismatches'
                                   % (index, probe, result['non_finite']))

    compare_scalars(ref_dir, test_dir, scene, tol, comparison)

    for name in ('A.mtx', 'b.mtx', 'x.mtx'):
        compare_mtx(name, ref_dir, test_dir, comparison,
                    effective_tolerance(tol, scene, name))

    summary = {
        'status': 'ok' if comparison.ok else 'violation',
        'scene': scene,
        'backend': test.get('backend'),
        'frames': test.get('frames'),
        'identical_frames': identical_frames,
        'total_frames': len(test_checksums),
        'violations': comparison.violations,
        'notes': comparison.notes,
    }
    if out_dir:
        write_reports(comparison, summary, out_dir)
    return comparison, summary


def write_reports(comparison, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'report.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'probe', 'geometry', 'count', 'max_abs', 'max_rel',
                         'count_over', 'worst_index', 'status'])
        for row in comparison.rows:
            writer.writerow([row['frame'], row['probe'], row['geometry'], row['count'],
                             '%.17g' % row['max_abs'], '%.17g' % row['max_rel'],
                             row['count_over'], row['worst_index'], row['status']])

    lines = []
    lines.append('# Parity report: %s' % summary.get('scene', '?'))
    lines.append('')
    lines.append('| Field | Value |')
    lines.append('|---|---|')
    lines.append('| status | **%s** |' % summary['status'])
    lines.append('| test backend | %s |' % summary.get('backend'))
    lines.append('| frames | %s |' % summary.get('frames'))
    lines.append('| frames with an identical checksum | %s/%s |'
                 % (summary['identical_frames'], summary['total_frames']))
    lines.append('| violations | %d |' % len(comparison.violations))
    lines.append('')
    if summary['identical_frames'] == summary['total_frames'] and comparison.ok:
        lines.append('Verdict: identical. Every frame checksum matches and every '
                     'element of every probe has delta == 0.0.')
    else:
        lines.append('Verdict: %s.' % ('VIOLATION' if not comparison.ok else 'in tolerance'))
    lines.append('')
    if comparison.violations:
        lines.append('## Violations')
        lines.append('')
        for text in comparison.violations[:20]:
            lines.append('- %s' % text)
        if len(comparison.violations) > 20:
            lines.append('- ... %d more (see report.csv)'
                         % (len(comparison.violations) - 20))
        lines.append('')
    if comparison.notes:
        lines.append('## Notes')
        lines.append('')
        for text in comparison.notes:
            lines.append('- %s' % text)
        lines.append('')
    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
#  --selftest: an .mtx that the parser re-serialises identically, plus a
#  negative control, so the Matrix-Market path is exercised without a CUDA stream
# ---------------------------------------------------------------------------
SELFTEST_MTX = """%%MatrixMarket matrix coordinate real general
% synthetic parity selftest
3 3 4
1 1 1.5
2 2 -2
3 3 0
1 3 1e-9
"""


def selftest():
    """Round-trip the Matrix-Market path on synthetic files (research §4 P3 exit)."""
    problems = []
    with tempfile.TemporaryDirectory(prefix='uipc_parity_selftest_') as tmp:
        ref_dir = os.path.join(tmp, 'ref')
        test_dir = os.path.join(tmp, 'test')
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, 'A.mtx')
        test_path = os.path.join(test_dir, 'A.mtx')

        def write(path, text):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)

        write(ref_path, SELFTEST_MTX)
        # Same entries in a different order: the comparison is order independent,
        # because both backends are only required to coalesce duplicates identically.
        write(test_path, '%%MatrixMarket matrix coordinate real general\n'
                         '% reordered\n3 3 4\n1 3 1e-9\n3 3 0\n2 2 -2\n1 1 1.5\n')
        comparison = Comparison()
        compare_mtx('A.mtx', ref_dir, test_dir, comparison, {'abs': 0.0, 'rel': 0.0})
        if not comparison.ok:
            problems.append('identical (reordered) .mtx files must compare clean: %r'
                            % comparison.violations)

        # negative control 1: one value moved by 1e-3 (Class A, so any delta is one)
        write(test_path, SELFTEST_MTX.replace('1.5', '1.501'))
        comparison = Comparison()
        compare_mtx('A.mtx', ref_dir, test_dir, comparison, {'abs': 0.0, 'rel': 0.0})
        if comparison.ok:
            problems.append('a 1e-3 perturbation must be a violation')

        # negative control 2: a sparsity difference (the pattern is a real bug gate)
        write(test_path, SELFTEST_MTX.replace('3 3 0\n', ''))
        comparison = Comparison()
        compare_mtx('A.mtx', ref_dir, test_dir, comparison, {'abs': 0.0, 'rel': 0.0})
        if comparison.ok:
            problems.append('a sparsity difference must be a violation')

        # negative control 3: duplicate coalescing (0.5 + 0.5 == 1.0, one cell)
        write(test_path, '%%MatrixMarket matrix coordinate real general\n3 3 2\n'
                         '1 1 0.5\n1 1 0.5\n')
        write(ref_path, '%%MatrixMarket matrix coordinate real general\n3 3 1\n'
                        '1 1 1.0\n')
        comparison = Comparison()
        compare_mtx('A.mtx', ref_dir, test_dir, comparison, {'abs': 0.0, 'rel': 0.0})
        if comparison.ok:
            problems.append('a duplicate-count difference must be a violation')

    if problems:
        for problem in problems:
            print('SELFTEST FAIL: %s' % problem, file=sys.stderr)
        return EXIT_VIOLATION
    print('SELFTEST OK: .mtx parse/compare round-trip is order independent, and a '
          '1e-3 perturbation, a sparsity difference and a duplicate-count difference '
          'are each a violation')
    return EXIT_OK


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare two parity artifacts (CUDA vs Luisa, or Luisa vs Luisa).')
    parser.add_argument('--ref', help='reference artifact directory (<out>/<backend>)')
    parser.add_argument('--test', help='test artifact directory (<out>/<backend>)')
    parser.add_argument('--out', help='directory for report.csv/report.md '
                                      '(default: the test directory)')
    parser.add_argument('--tolerances', default=DEFAULT_TOLERANCES,
                        help='tolerances.json (default: apps/tests/parity/'
                             'tolerances.json)')
    parser.add_argument('--check-tolerances', action='store_true',
                        help='only audit the tolerance file and exit')
    parser.add_argument('--selftest', action='store_true',
                        help='exercise the Matrix-Market path on synthetic files')
    parser.add_argument('--quiet', action='store_true', help='only print the verdict')
    args = parser.parse_args(argv)

    tol = load_tolerances(args.tolerances)

    if args.check_tolerances:
        problems = check_tolerances(tol)
        for problem in problems:
            print('tolerances: %s' % problem, file=sys.stderr)
        if problems:
            print('FAIL %d tolerance(s) without a _why entry' % len(problems))
            return EXIT_VIOLATION
        print('OK %s: every non-default tolerance carries a _why entry'
              % (args.tolerances or '<builtin>'))
        return EXIT_OK

    if args.selftest:
        return selftest()

    if not args.ref or not args.test:
        parser.error('--ref and --test are required (or use --check-tolerances / '
                     '--selftest)')

    for directory in (args.ref, args.test):
        if not os.path.isdir(directory):
            print('missing directory %s' % directory, file=sys.stderr)
            return EXIT_USAGE

    out_dir = args.out if args.out else args.test
    comparison, summary = compare_dirs(args.ref, args.test, tol, out_dir)

    if summary['status'] == 'skipped':
        print('SKIP: %s (%s side)' % (summary['reason'], summary.get('side')))
        return EXIT_SKIPPED
    if summary['status'] == 'error':
        for text in summary.get('violations', []):
            print(text, file=sys.stderr)
        return EXIT_VIOLATION

    if not args.quiet:
        for text in comparison.violations[:20]:
            print(text)
        if len(comparison.violations) > 20:
            print('... %d more violations (see %s)'
                  % (len(comparison.violations) - 20,
                     os.path.join(out_dir, 'report.csv')))
        for text in comparison.notes:
            print(text)
        print('frames with an identical checksum: %d/%d'
              % (summary['identical_frames'], summary['total_frames']))
        worst = max((row['max_abs'] for row in comparison.rows), default=0.0)
        print('max |delta| over every compared element: %.17g' % worst)

    if not comparison.ok:
        print('FAIL: %d violation(s); report written to %s'
              % (len(comparison.violations), os.path.join(out_dir, 'report.md')))
        return EXIT_VIOLATION
    print('OK: %s matches %s (delta == 0.0 on every element%s)'
          % (args.test, args.ref,
             ', %d/%d frames identical' % (summary['identical_frames'],
                                           summary['total_frames'])
             if summary['total_frames'] else ''))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
