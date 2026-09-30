"""Manifold covariance estimation and portfolio risk under synthetic regimes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize

from .experiment import HERE
from .spd import distance, power, geodesic, log_euclidean_mean, frechet_mean

PROTOCOL = {
    'seed': 20261001, 'paths': 20, 'sessions': 1008, 'assets': 12,
    'training_sessions': 252, 'validation_sessions': 63, 'covariance_block': 63,
    'rebalance_sessions': 21, 'regime_breaks': [0, 336, 672],
    'student_degrees_of_freedom': 5,
    'shrinkage_grid': [0., .05, .15, .35, .65],
    'position_cap': .25, 'one_way_cost_bp': 5., 'annual_sessions': 252,
    'bootstrap_draws': 4000,
    'selection': 'initial chronological validation only; per-method alpha then frozen throughout the holdout',
    'covariance_target': 'diagonal of each estimated covariance',
    'score': 'zero-mean Gaussian quasi-likelihood, constants omitted, per asset',
    'scope': 'synthetic zero-mean Student t returns; known regime covariances; daily weight drift and proportional rebalance costs',
}


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def covariance_estimate(returns, method, alpha=0., block=63):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or len(returns) < 2 or not np.isfinite(returns).all() or not 0 <= alpha <= 1:
        raise ValueError('Finite returns and an intensity in [0, 1] are required.')
    if method in ['sample', 'shrinkage']:
        estimate = np.cov(returns, rowvar=False, ddof=1)
        diagnostic = {'iterations': 0, 'gradient_norm': 0., 'objective': 0., 'converged': True}
    else:
        if len(returns) % block:
            raise ValueError('Block estimators require complete equally sized blocks.')
        matrices = np.array([np.cov(piece, rowvar=False, ddof=1) for piece in np.split(returns, len(returns)//block)])
        # Positive definiteness comes from sample size, never from a silent eigenvalue floor.
        if method == 'log_mean':
            estimate = log_euclidean_mean(matrices)
            diagnostic = {'iterations': 0, 'gradient_norm': 0., 'objective': 0., 'converged': True}
        elif method == 'affine_mean':
            estimate, diagnostic = frechet_mean(matrices)
        else:
            raise ValueError(f'Unknown estimator: {method}')
    estimate = (1-alpha)*estimate+alpha*np.diag(np.diag(estimate))
    power(estimate, 0)
    return estimate, diagnostic


def quasi_likelihood(returns, covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError('Covariance must be positive definite.')
    quadratic = np.sum(returns*np.linalg.solve(covariance, returns.T).T, axis=1)
    return (logdet+quadratic)/covariance.shape[0]


def minimum_variance(covariance, cap=.25):
    dimension = covariance.shape[0]
    if cap*dimension < 1:
        raise ValueError('The position cap makes full investment infeasible.')
    scale = np.trace(covariance)/dimension
    normalized = covariance/scale
    result = minimize(lambda w: float(w @ normalized @ w), np.full(dimension, 1/dimension),
                      jac=lambda w: 2*normalized @ w, method='SLSQP', bounds=[(0., cap)]*dimension,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1,
                                    'jac': lambda w: np.ones(dimension)}],
                      options={'ftol': 1e-12, 'maxiter': 200})
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x
    if abs(weights.sum()-1) > 1e-8 or weights.min() < -1e-8 or weights.max() > cap+1e-8:
        raise RuntimeError('Portfolio constraints failed.')
    return weights


def synthetic_path(rng, sessions, assets):
    """Three reproducible covariance regimes, with variance-matched t shocks."""
    loadings = rng.normal(0, .004, size=(assets, 3))
    idiosyncratic = np.linspace(.006, .014, assets)**2
    first = loadings @ loadings.T+np.diag(idiosyncratic)
    rotation, _ = np.linalg.qr(rng.normal(size=(assets, assets)))
    second = rotation @ first @ rotation.T*2.0
    third = first*.6+np.outer(np.linspace(-.008, .008, assets), np.linspace(-.008, .008, assets))
    regimes = np.array([first, second, third])
    state = np.minimum(np.arange(sessions)//336, 2)
    nu = PROTOCOL['student_degrees_of_freedom']
    returns = np.empty((sessions, assets))
    for index, covariance in enumerate(regimes):
        mask = state == index
        gaussian = rng.normal(size=(mask.sum(), assets)) @ np.linalg.cholesky(covariance).T
        returns[mask] = gaussian*np.sqrt((nu-2)/rng.chisquare(nu, mask.sum()))[:, None]
    if np.min(returns) <= -.95:
        raise RuntimeError('Synthetic simple returns are outside the specified accounting domain.')
    return returns, regimes, state


def paired_path_interval(baseline, candidate, rng, draws):
    indices = rng.integers(0, len(baseline), size=(draws, len(baseline)))
    reductions = 1-np.sqrt(np.mean(candidate[indices], axis=1)/np.mean(baseline[indices], axis=1))
    return np.quantile(reductions, [.025, .975]).tolist()


def portfolio_step(weights, returns, cost=0.):
    """Pay a proportional cost before returns, then drift unchanged holdings."""
    if not 0 <= cost < 1 or np.min(returns) <= -1:
        raise ValueError('Accounting requires cost in [0, 1) and returns greater than -1.')
    gross = float(weights @ returns)
    net = (1-cost)*(1+gross)-1
    drifted = weights*(1+returns)/(1+gross)
    return gross, net, drifted


def run(output=None, quick=False):
    out = Path(output) if output else HERE/'results/research'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    if quick:
        protocol.update(paths=2, sessions=336, bootstrap_draws=200)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    methods = ['sample', 'shrinkage', 'log_mean', 'affine_mean']
    strategies = ['equal_weight', *methods, 'oracle']
    streams = np.random.SeedSequence(protocol['seed']).spawn(protocol['paths']+1)
    portfolio_rows, forecast_rows, selection_rows, summary_rows, return_rows = [], [], [], [], []
    all_covariances = []
    largest_gradient, maximum_iterations, mean_computations = 0., 0, 0
    start = protocol['training_sessions']
    for path in range(protocol['paths']):
        returns, covariances, state = synthetic_path(np.random.default_rng(streams[path]), protocol['sessions'], protocol['assets'])
        all_covariances.append(covariances)
        for session, values in enumerate(returns):
            return_rows.append({'path': path, 'session': session, 'regime': int(state[session]),
                                **{f'asset_{i:02d}': value for i, value in enumerate(values)}})
        fit_end = start-protocol['validation_sessions']
        selected = {'sample': 0.}
        for method in methods[1:]:
            base, _ = covariance_estimate(returns[:fit_end], method, block=protocol['covariance_block'])
            scores = []
            for alpha in protocol['shrinkage_grid']:
                covariance = (1-alpha)*base+alpha*np.diag(np.diag(base))
                score = float(quasi_likelihood(returns[fit_end:start], covariance).mean())
                scores.append((score, alpha))
                selection_rows.append({'path': path, 'method': method, 'alpha': alpha,
                                       'fit_last_session': fit_end-1, 'validation_last_session': start-1,
                                       'validation_quasi_likelihood': score})
            selected[method] = min(scores)[1]
        current_weights = {name: np.full(protocol['assets'], 1/protocol['assets']) for name in strategies}
        values = {name: [] for name in strategies}
        net_values = {name: [] for name in strategies}
        cost_totals = dict.fromkeys(strategies, 0.)
        active_covariances = {}
        for session in range(start, protocol['sessions']):
            costs = dict.fromkeys(strategies, 0.)
            if (session-start) % protocol['rebalance_sessions'] == 0:
                history = returns[session-start:session]
                for method in methods:
                    covariance, diag = covariance_estimate(history, method, selected[method], protocol['covariance_block'])
                    active_covariances[method] = covariance
                    if method == 'affine_mean':
                        largest_gradient = max(largest_gradient, diag['gradient_norm'])
                        maximum_iterations = max(maximum_iterations, diag['iterations'])
                        mean_computations += 1
                    oracle = covariances[state[session]]
                    forecast_rows.append({'path': path, 'session': session, 'last_fit_session': session-1,
                                          'method': method, 'selected_alpha': selected[method],
                                          'affine_distance_to_true': distance(covariance, oracle),
                                          'condition_number': float(np.linalg.cond(covariance)),
                                          'gradient_norm': diag['gradient_norm'], 'mean_iterations': diag['iterations']})
                active_covariances['oracle'] = covariances[state[session]]
                active_covariances['equal_weight'] = active_covariances['sample']
                for name in strategies:
                    target_weights = np.full(protocol['assets'], 1/protocol['assets']) if name == 'equal_weight' else minimum_variance(active_covariances[name], protocol['position_cap'])
                    turnover = float(np.abs(target_weights-current_weights[name]).sum())
                    costs[name] = protocol['one_way_cost_bp']/10000*turnover
                    current_weights[name] = target_weights
            for name in strategies:
                weights = current_weights[name]
                gross, net, drifted = portfolio_step(weights, returns[session], costs[name])
                prediction = float(weights @ active_covariances[name] @ weights)
                true_variance = float(weights @ covariances[state[session]] @ weights)
                score = float(quasi_likelihood(returns[session:session+1], active_covariances[name])[0])
                portfolio_rows.append({'path': path, 'session': session, 'regime': int(state[session]),
                                       'strategy': name, 'gross_return': gross, 'net_return': net,
                                       'cost_fraction': costs[name], 'predicted_variance': prediction,
                                       'true_variance': true_variance, 'quasi_likelihood': score,
                                       'max_weight_before_return': float(weights.max())})
                values[name].append(gross)
                net_values[name].append(net)
                cost_totals[name] += costs[name]
                # Self-financing holdings drift after every simple return.
                current_weights[name] = drifted
        for name in strategies:
            gross, net = np.array(values[name]), np.array(net_values[name])
            wealth = np.r_[1., np.cumprod(1+net)]
            drawdown = wealth/np.maximum.accumulate(wealth)-1
            tail_count = max(1, int(np.ceil(.01*len(gross))))
            summary_rows.append({'path': path, 'strategy': name, 'observations': len(gross),
                                 'mean_squared_return': float(np.mean(gross**2)),
                                 'annualized_volatility': float(np.std(gross, ddof=1)*np.sqrt(252)),
                                 'es99_loss': float(np.sort(-gross)[-tail_count:].mean()),
                                 'net_terminal_wealth': float(wealth[-1]), 'maximum_drawdown': float(-drawdown.min()),
                                 'total_cost_fraction': cost_totals[name]})
        print(f'Covariance path {path+1}/{protocol["paths"]}: selected shrinkages={selected}', flush=True)
    write_csv(out/'synthetic_returns.csv', return_rows)
    np.savez_compressed(out/'true_covariances.npz', covariances=np.array(all_covariances))
    write_csv(out/'daily_portfolios.csv', portfolio_rows)
    write_csv(out/'covariance_diagnostics.csv', forecast_rows)
    write_csv(out/'model_selection.csv', selection_rows)
    write_csv(out/'path_metrics.csv', summary_rows)
    bootstrap_rng = np.random.default_rng(streams[-1])
    baseline = np.array([row['mean_squared_return'] for row in summary_rows if row['strategy'] == 'sample'])
    comparisons, aggregate = [], []
    for name in strategies:
        subset = [row for row in summary_rows if row['strategy'] == name]
        mse = np.array([row['mean_squared_return'] for row in subset])
        interval = paired_path_interval(baseline, mse, bootstrap_rng, protocol['bootstrap_draws'])
        comparisons.append({'strategy': name, 'rms_reduction_vs_sample': float(1-np.sqrt(mse.mean()/baseline.mean())),
                            'ci95_low': interval[0], 'ci95_high': interval[1]})
        daily = [row for row in portfolio_rows if row['strategy'] == name]
        aggregate.append({'strategy': name, 'paths': protocol['paths'],
                          'mean_annualized_volatility': float(np.mean([row['annualized_volatility'] for row in subset])),
                          'mean_es99_loss': float(np.mean([row['es99_loss'] for row in subset])),
                          'mean_total_cost_fraction': float(np.mean([row['total_cost_fraction'] for row in subset])),
                          'mean_quasi_likelihood': float(np.mean([row['quasi_likelihood'] for row in daily])),
                          'variance_forecast_relative_bias': float(np.mean([row['predicted_variance'] for row in daily])/np.mean([row['true_variance'] for row in daily])-1)})
    write_csv(out/'paired_path_intervals.csv', comparisons)
    write_csv(out/'aggregate_metrics.csv', aggregate)
    diagnostics = {'affine_mean_computations': mean_computations, 'maximum_gradient_norm': largest_gradient,
                   'maximum_mean_iterations': maximum_iterations,
                   'all_forecasts_use_past_returns': all(row['last_fit_session'] < row['session'] for row in forecast_rows),
                   'oracle_covariance_is_unobservable': True}
    metrics = {'protocol': protocol, 'aggregate': aggregate, 'comparisons': comparisons,
               'diagnostics': diagnostics,
               'code_sha256': {str(p.relative_to(HERE)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((HERE/'src/geometry_lab').glob('*.py'))},
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.suffix in ['.csv', '.npz']}}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [row['strategy'] for row in aggregate]
    axes[0, 0].bar(labels, [100*row['mean_annualized_volatility'] for row in aggregate])
    axes[0, 0].set(title='Synthetic holdout portfolio risk', ylabel='Mean annualized volatility, %')
    x = np.arange(len(comparisons))
    y = np.array([row['rms_reduction_vs_sample'] for row in comparisons])
    lo = np.array([row['ci95_low'] for row in comparisons])
    hi = np.array([row['ci95_high'] for row in comparisons])
    axes[0, 1].vlines(x, lo*100, hi*100)
    axes[0, 1].scatter(x, y*100)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].axhline(0, color='black', lw=.7)
    axes[0, 1].set(title='Paired path bootstrap: pointwise 95%', ylabel='RMS reduction vs sample, %')
    for method in methods:
        sessions = sorted({row['session'] for row in forecast_rows})
        errors = [np.mean([row['affine_distance_to_true'] for row in forecast_rows if row['session'] == session and row['method'] == method]) for session in sessions]
        axes[1, 0].plot(sessions, errors, label=method)
    for point in protocol['regime_breaks'][1:]:
        axes[1, 0].axvline(point, color='grey', ls='--', lw=.7)
    axes[1, 0].set(title='Covariance error and regime transitions', ylabel='Affine-invariant distance', xlabel='Session')
    axes[1, 0].legend(fontsize=8)
    a, b = np.diag([1., 9.]), np.diag([9., 1.])
    ts = np.linspace(0, 1, 101)
    axes[1, 1].plot(ts, [np.linalg.det((1-t)*a+t*b) for t in ts], label='Arithmetic path')
    axes[1, 1].plot(ts, [np.linalg.det(geodesic(a, b, t)) for t in ts], label='Affine geodesic')
    axes[1, 1].set(title='Determinant along two interpolation paths', xlabel='t', ylabel='determinant')
    axes[1, 1].legend(fontsize=8)
    for axis in axes[0]:
        axis.tick_params(axis='x', labelrotation=25)
    fig.tight_layout()
    fig.savefig(out/'covariance_research.png', dpi=180)
    fig.savefig(out/'covariance_research.pdf')
    plt.close(fig)
    print(json.dumps({'aggregate': aggregate, 'diagnostics': diagnostics}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run(args.output, args.quick)


if __name__ == '__main__':
    main()
