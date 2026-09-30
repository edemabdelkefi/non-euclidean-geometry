from pathlib import Path
import csv
import hashlib
import json
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/research'


class ResultTests(unittest.TestCase):
    def test_frozen_artifact_checksums(self):
        metrics = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        for name, digest in metrics['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/name).read_bytes()).hexdigest(), digest, name)
        for name, digest in metrics['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), digest, name)

    def test_self_financing_paths_reconstructed(self):
        with (OUT/'daily_portfolios.csv').open(encoding='utf-8', newline='') as handle:
            daily = list(csv.DictReader(handle))
        grouped = {}
        for row in daily:
            key = (int(row['path']), row['strategy'])
            grouped.setdefault(key, []).append(row)
            expected = (1-float(row['cost_fraction']))*(1+float(row['gross_return']))-1
            self.assertAlmostEqual(float(row['net_return']), expected, places=12)
        with (OUT/'path_metrics.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                group = grouped[int(row['path']), row['strategy']]
                returns = np.array([float(day['net_return']) for day in group])
                self.assertAlmostEqual(np.prod(1+returns), float(row['net_terminal_wealth']), places=10)
                gross = np.array([float(day['gross_return']) for day in group])
                self.assertAlmostEqual(np.mean(gross**2), float(row['mean_squared_return']), places=12)
