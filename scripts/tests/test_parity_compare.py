#!/usr/bin/env python3
"""Unit tests for scripts/parity_compare.py (S0 / T9.3a).

Synthetic fixtures only: this is the one place the Class-B scalar path, the
checksum verification and the Matrix-Market path are exercised, because no CUDA
stream exists on this machine (research §5 of 03-parity-harness-research.md).

Run with:
    python -m unittest scripts.tests.test_parity_compare -v
"""
import array
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
COMPARE = os.path.join(SCRIPTS, 'parity_compare.py')
TOLERANCES = os.path.join(ROOT, 'apps', 'tests', 'parity', 'tolerances.json')

sys.path.insert(0, SCRIPTS)
import parity_compare  # noqa: E402  (after sys.path setup)


def write_manifest(directory, frames, arrays, checksums, scene='synthetic',
                   scene_hash=1.0, backend='luisa'):
    manifest = {
        'schema': 'uipc-parity/1',
        'backend': backend,
        'scene': scene,
        'scene_hash': scene_hash,
        'frames': frames,
        'dt': 0.01,
        'arrays': arrays,
        'checksums': checksums,
        'scalars': {'file': 'scalars.jsonl', 'frames': frames},
    }
    with open(os.path.join(directory, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def write_f64(path, values):
    with open(path, 'wb') as f:
        f.write(struct.pack('<%dd' % len(values), *values))


def write_scalars(directory, stats_per_frame):
    with open(os.path.join(directory, 'scalars.jsonl'), 'w', encoding='utf-8') as f:
        for index, stats in enumerate(stats_per_frame):
            f.write(json.dumps({'frame': index, 'stats': stats}) + '\n')


def make_leg(directory, values_per_frame, stats_per_frame, scene='synthetic'):
    """One synthetic artifact: N frames, one `vertices` probe of 3 components."""
    os.makedirs(directory, exist_ok=True)
    arrays = []
    for index, values in enumerate(values_per_frame):
        name = 'frame_%03d_vertices_g0.f64' % index
        write_f64(os.path.join(directory, name), values)
        arrays.append([{'probe': 'vertices', 'geometry': 0, 'dtype': 'f64',
                        'components': 3, 'count': len(values) // 3, 'file': name}])
    checksums = [{'frame': index,
                  'value': parity_compare.frame_checksum(frame_arrays, directory)}
                 for index, frame_arrays in enumerate(arrays)]
    write_manifest(directory, len(arrays), arrays, checksums, scene=scene)
    write_scalars(directory, stats_per_frame)


def stats(newton=3, converged=True, sync_ms=0.02):
    return {
        'backend': 'luisa',
        'frame': 1,
        'pipeline': 'ipc',
        'completed': True,
        'converged': converged,
        'newton_iterations': newton,
        'line_search_trials': 1,
        'linear_solver_iterations': 12,
        'hit_newton_limit': False,
        'hit_line_search_limit': False,
        'last_line_search_alpha': 0.5,
        'last_ccd_toi': 1.0,
        'last_cfl_alpha': 1.0,
        'counters': {'commits': 1, 'commands': 4, 'compiles': 0, 'sync_ms': sync_ms},
    }


class ComparatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='uipc_parity_test_')
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def leg(self, name):
        return os.path.join(self.root, name)

    def run_compare(self, ref, test, *extra):
        out = os.path.join(self.root, 'report')
        command = [sys.executable, COMPARE, '--ref', ref, '--test', test, '--out', out]
        command.extend(extra)
        completed = subprocess.run(command, capture_output=True, text=True)
        return completed.returncode, completed.stdout + completed.stderr, out

    # -- the seed cell: a self-comparison must be exact --------------------
    def test_self_comparison_is_exact(self):
        values = [[0.0, -1.5, 2.25, 1e-30, -0.0, 3.0]] * 2
        make_leg(self.leg('a'), values, [stats(), stats()])
        make_leg(self.leg('b'), values, [stats(), stats()])
        rc, text, out = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 0, text)
        self.assertIn('delta == 0.0', text)
        self.assertTrue(os.path.isfile(os.path.join(out, 'report.csv')))
        self.assertTrue(os.path.isfile(os.path.join(out, 'report.md')))

    def test_one_ulp_drift_is_a_violation(self):
        import math
        base = [0.0, 1.5, -2.25, 4.0, 5.5, 6.75]
        drifted = list(base)
        drifted[3] = math.nextafter(drifted[3], math.inf)
        make_leg(self.leg('a'), [base], [stats()])
        make_leg(self.leg('b'), [drifted], [stats()])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        # the failure must name the probe and the element index
        self.assertIn('vertices[3]', text)

    def test_shape_mismatch_is_a_violation(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        make_leg(self.leg('b'), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], [stats()])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)

    # -- Class B: structural scalar comparison ----------------------------
    def test_structurally_different_scalars_fail(self):
        values = [[0.0, 1.0, 2.0]]
        make_leg(self.leg('a'), values, [stats(newton=3)])
        make_leg(self.leg('b'), values, [stats(newton=4)])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        self.assertIn('newton_iterations', text)

    def test_flag_difference_fails(self):
        values = [[0.0, 1.0, 2.0]]
        make_leg(self.leg('a'), values, [stats(converged=True)])
        make_leg(self.leg('b'), values, [stats(converged=False)])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        self.assertIn('converged', text)

    def test_wall_clock_scalar_is_ignored(self):
        values = [[0.0, 1.0, 2.0]]
        make_leg(self.leg('a'), values, [stats(sync_ms=0.02)])
        make_leg(self.leg('b'), values, [stats(sync_ms=0.51)])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 0, text)

    # -- the checksum gate must not trust the manifest --------------------
    def test_tampered_array_is_caught_even_if_the_manifest_agrees(self):
        values = [[0.0, 1.0, 2.0]]
        make_leg(self.leg('a'), values, [stats()])
        make_leg(self.leg('b'), values, [stats()])
        # Change a value on the test side without touching its recorded checksum: the
        # recomputation from the raw arrays is the only thing that can catch this.
        write_f64(os.path.join(self.leg('b'), 'frame_000_vertices_g0.f64'),
                  [0.0, 1.0, 9.0])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        self.assertIn('checksum recorded', text)

    # -- the SKIP contract ------------------------------------------------
    def test_skip_json_returns_3(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        os.makedirs(self.leg('b'))
        with open(os.path.join(self.leg('b'), 'SKIP.json'), 'w', encoding='utf-8') as f:
            json.dump({'schema': 'uipc-parity-skip/1', 'status': 'skipped',
                       'backend': 'cuda',
                       'reason': 'uipc_backend_cuda.dll not found'}, f)
        rc, text, out = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 3, text)
        self.assertIn('SKIP', text)
        self.assertTrue(os.path.isfile(os.path.join(out, 'report.md')))

    def test_missing_manifest_is_a_violation(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        os.makedirs(self.leg('b'))
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)

    def test_scene_hash_mismatch_is_a_violation(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        make_leg(self.leg('b'), [[0.0, 1.0, 2.0]], [stats()])
        manifest_path = os.path.join(self.leg('b'), 'manifest.json')
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        manifest['scene_hash'] = 42.0
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        self.assertIn('scene_hash', text)

    # -- the tolerance file is self-auditing ------------------------------
    def test_tolerances_without_why_fail(self):
        with open(TOLERANCES, encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault('probes', {})['vertices'] = {'rel': 1e-9}
        path = os.path.join(self.root, 'bad_tolerances.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        completed = subprocess.run(
            [sys.executable, COMPARE, '--check-tolerances', '--tolerances', path],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn('probes.vertices.rel', completed.stdout + completed.stderr)

    def test_committed_tolerances_are_clean(self):
        completed = subprocess.run(
            [sys.executable, COMPARE, '--check-tolerances', '--tolerances', TOLERANCES],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_a_why_entry_satisfies_the_audit(self):
        with open(TOLERANCES, encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault('probes', {})['vertices'] = {'rel': 1e-9}
        data.setdefault('_why', {})['probes.vertices.rel'] = 'synthetic test entry'
        path = os.path.join(self.root, 'good_tolerances.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        completed = subprocess.run(
            [sys.executable, COMPARE, '--check-tolerances', '--tolerances', path],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_tolerance_is_used_for_the_comparison(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        make_leg(self.leg('b'), [[0.0, 1.0, 2.0001]], [stats()])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)  # bit-exact by default
        path = os.path.join(self.root, 'loose.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'schema': 'uipc-parity-tolerances/1',
                       'probes': {'vertices': {'rel': 1e-3}},
                       '_why': {'probes.vertices.rel': 'synthetic'}}, f)
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'),
                                       '--tolerances', path)
        self.assertEqual(rc, 0, text)

    # -- Matrix Market ----------------------------------------------------
    def test_mtx_selftest(self):
        completed = subprocess.run([sys.executable, COMPARE, '--selftest'],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('SELFTEST OK', completed.stdout)

    def test_absent_linear_system_is_not_a_failure(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        make_leg(self.leg('b'), [[0.0, 1.0, 2.0]], [stats()])
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 0, text)
        self.assertIn('A.mtx: absent', text)

    def test_matrix_market_values_are_compared(self):
        make_leg(self.leg('a'), [[0.0, 1.0, 2.0]], [stats()])
        make_leg(self.leg('b'), [[0.0, 1.0, 2.0]], [stats()])
        for directory, scale in ((self.leg('a'), 1.0), (self.leg('b'), 1.0)):
            with open(os.path.join(directory, 'A.mtx'), 'w', encoding='utf-8') as f:
                f.write('%%MatrixMarket matrix coordinate real general\n2 2 2\n')
                f.write('1 1 %.17g\n2 2 %.17g\n' % (1.5 * scale, -2.5 * scale))
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 0, text)
        with open(os.path.join(self.leg('b'), 'A.mtx'), 'w', encoding='utf-8') as f:
            f.write('%%MatrixMarket matrix coordinate real general\n2 2 2\n')
            f.write('1 1 1.5\n2 2 -2.5000001\n')
        rc, text, _ = self.run_compare(self.leg('a'), self.leg('b'))
        self.assertEqual(rc, 1, text)
        self.assertIn('A.mtx', text)


if __name__ == '__main__':
    unittest.main()
