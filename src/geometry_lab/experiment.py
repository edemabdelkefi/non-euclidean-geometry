from pathlib import Path
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results"
SEED = 20260930

PROTOCOL = {'seed': 20260930, 'geometry': {'geodesic_pairs': 1000, 'disk_radius': 0.75, 'simpson_intervals': [16, 32, 64, 128, 256, 512], 'mobius_cases': 10000, 'lorentz_cases': 10000, 'beta_range': [-0.99, 0.99]}}

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def simpson(values):
    intervals = values.shape[-1] - 1
    assert intervals % 2 == 0
    return (values[..., 0] + values[..., -1] + 4 * values[..., 1:-1:2].sum(axis=-1) + 2 * values[..., 2:-1:2].sum(axis=-1)) / (3 * intervals)


def disk_points(rng, count, radius):
    return radius * np.sqrt(rng.uniform(size=count)) * np.exp(2j * np.pi * rng.uniform(size=count))


def geodesic(a, b, samples):
    w = (b - a) / (1 - np.conj(a) * b)
    s = np.linspace(0, 1, samples + 1)[None, :]
    denominator = 1 + np.conj(a[:, None]) * s * w[:, None]
    z = (a[:, None] + s * w[:, None]) / denominator
    dz = w[:, None] * (1 - np.abs(a[:, None])**2) / denominator**2
    metric_speed = 2 * np.abs(dz) / (1 - np.abs(z)**2)
    return simpson(metric_speed), z


def cross_ratio(z):
    return (z[:, 0] - z[:, 2]) * (z[:, 1] - z[:, 3]) / ((z[:, 0] - z[:, 3]) * (z[:, 1] - z[:, 2]))


def geometry_experiments(streams):
    spec, rng = PROTOCOL['geometry'], streams[0]
    count = spec['geodesic_pairs']
    a = disk_points(rng, count, spec['disk_radius'])
    b = disk_points(rng, count, spec['disk_radius'])
    w = (b - a)/(1 - np.conj(a)*b)
    exact = 2 * np.arctanh(np.abs(w))
    assert np.all(exact > 0)
    errors, rows = [], []
    for intervals in spec['simpson_intervals']:
        lengths, paths = geodesic(a, b, intervals)
        assert np.max(np.abs(paths[:, 0] - a)) < 1e-14
        assert np.max(np.abs(paths[:, -1] - b)) < 1e-14
        assert np.max(np.abs(paths)) < 1
        rel = np.abs(lengths - exact)/exact
        errors.append(dict(intervals=intervals, max_relative_error=float(rel.max()), median_relative_error=float(np.median(rel)), rmse=float(np.sqrt(np.mean((lengths-exact)**2)))))
        for i in range(count):
            rows.append(dict(case=i, intervals=intervals, a_real=a[i].real, a_imag=a[i].imag, b_real=b[i].real, b_imag=b[i].imag, exact_length=exact[i], integrated_length=lengths[i], relative_error=rel[i]))
    order = float(-np.polyfit(np.log([e['intervals'] for e in errors][-4:]), np.log([e['max_relative_error'] for e in errors][-4:]), 1)[0])
    write_csv('geodesics.csv', rows)
    write_csv('geodesic_convergence.csv', errors)

    rng = streams[1]
    count = spec['mobius_cases']
    z = disk_points(rng, count*4, 1.).reshape(count, 4)
    matrices = []
    transformed = []
    attempts = 0
    for points in z:
        while True:
            attempts += 1
            m = rng.normal(size=4) + 1j * rng.normal(size=4)
            m /= np.linalg.norm(m)
            aa, bb, cc, dd = m
            # Explicit conditioning restrictions; no rejection based on measured error.
            if abs(aa*dd-bb*cc) >= .15 and np.min(np.abs(cc*points+dd)) >= .05:
                matrices.append(m)
                transformed.append((aa*points+bb)/(cc*points+dd))
                break
    cr, cr_after = cross_ratio(z), cross_ratio(np.array(transformed))
    cr_error = np.abs(cr_after-cr)/np.maximum(1, np.abs(cr))
    write_csv('mobius.csv', [dict(case=i, normalized_error=cr_error[i], original_real=cr[i].real, original_imag=cr[i].imag, transformed_real=cr_after[i].real, transformed_imag=cr_after[i].imag) for i in range(count)])

    rng, count = streams[2], spec['lorentz_cases']
    events = rng.normal(size=(count, 4))  # column order: ct, x, y, z
    beta = rng.uniform(*spec['beta_range'], size=count)
    gamma = 1 / np.sqrt(1-beta**2)
    boosted = events.copy()
    boosted[:, 0] = gamma*(events[:, 0]-beta*events[:, 1])
    boosted[:, 1] = gamma*(events[:, 1]-beta*events[:, 0])
    invariant = -events[:, 0]**2 + (events[:, 1:]**2).sum(axis=1)
    after = -boosted[:, 0]**2 + (boosted[:, 1:]**2).sum(axis=1)
    invariant_error = np.abs(after-invariant)/np.maximum(1, np.abs(invariant))
    write_csv('lorentz.csv', [dict(case=i, beta=beta[i], invariant=invariant[i], boosted_invariant=after[i], normalized_error=invariant_error[i]) for i in range(count)])
    assert errors[-1]['max_relative_error'] < errors[0]['max_relative_error']
    assert cr_error.max() < 1e-10
    assert invariant_error.max() < 1e-10

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    theta = np.linspace(0, 2*np.pi, 500)
    ax[0].plot(np.cos(theta), np.sin(theta), 'k-', lw=1)
    for i in range(6):
        _, path = geodesic(a[i:i+1], b[i:i+1], 128)
        ax[0].plot(path[0].real, path[0].imag)
        ax[0].plot([a[i].real, b[i].real], [a[i].imag, b[i].imag], 'k.', ms=4)
    ax[0].set_aspect('equal')
    ax[0].set(title='Poincare-disk geodesics', xlabel='Re(z)', ylabel='Im(z)')
    ax[1].loglog([e['intervals'] for e in errors], [e['max_relative_error'] for e in errors], 'o-', label='Maximum over 1000 pairs')
    ax[1].set(title=f'Simpson convergence: order {order:.2f}', xlabel='Integration intervals', ylabel='Maximum relative length error')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'geometry_benchmarks.png', dpi=180)
    fig.savefig(OUT / 'geometry_benchmarks.pdf')
    plt.close(fig)
    metrics = dict(geodesic_pairs=spec['geodesic_pairs'], simpson_intervals=512, max_relative_geodesic_error=errors[-1]['max_relative_error'], convergence_order=order, mobius_cases=spec['mobius_cases'], mobius_max_normalized_error=float(cr_error.max()), mobius_candidate_matrices=attempts, lorentz_cases=count, lorentz_max_normalized_error=float(invariant_error.max()))
    print('Geometry experiments completed: ' + json.dumps(metrics), flush=True)
    return metrics


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(7)][4:]
    metrics = geometry_experiments(streams)
    results = {"protocol": PROTOCOL, "geometry": metrics, "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    results["csv_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
