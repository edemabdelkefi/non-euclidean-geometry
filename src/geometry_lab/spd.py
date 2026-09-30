"""Affine-invariant geometry on symmetric positive definite matrices."""
import numpy as np


def symmetric(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('A finite square matrix is required.')
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError('The matrix must be symmetric.')
    return (matrix+matrix.T)/2


def spectral(matrix, function, positive=True):
    values, vectors = np.linalg.eigh(symmetric(matrix))
    if positive and np.min(values) <= 0:
        raise ValueError('The matrix must be positive definite.')
    result = (vectors*function(values)) @ vectors.T
    return (result+result.T)/2


def power(matrix, exponent):
    return spectral(matrix, lambda values: values**exponent)


def matrix_log(matrix):
    return spectral(matrix, np.log)


def matrix_exp(matrix):
    return spectral(matrix, np.exp, positive=False)


def whiten(base, matrix):
    inverse_root = power(base, -.5)
    result = inverse_root @ matrix @ inverse_root
    return (result+result.T)/2


def distance(a, b):
    return float(np.linalg.norm(matrix_log(whiten(a, b)), 'fro'))


def geodesic(a, b, time):
    root = power(a, .5)
    result = root @ power(whiten(a, b), time) @ root
    return (result+result.T)/2


def log_map(base, point):
    root = power(base, .5)
    result = root @ matrix_log(whiten(base, point)) @ root
    return (result+result.T)/2


def exp_map(base, tangent):
    root = power(base, .5)
    result = root @ matrix_exp(whiten(base, tangent)) @ root
    return (result+result.T)/2


def inner_product(base, first, second):
    inverse = np.linalg.inv(symmetric(base))
    return float(np.trace(inverse @ first @ inverse @ second))


def parallel_transport(a, b, tangent):
    """Transport along the unique affine-invariant geodesic from a to b."""
    root = power(a, .5)
    inverse_root = power(a, -.5)
    transform = root @ power(whiten(a, b), .5) @ inverse_root
    result = transform @ symmetric(tangent) @ transform.T
    return (result+result.T)/2


def normalized_weights(count, weights=None):
    if count < 1:
        raise ValueError('At least one matrix is required.')
    weights = np.full(count, 1/count) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Nonnegative weights with positive total mass are required.')
    return weights/weights.sum()


def log_euclidean_mean(matrices, weights=None):
    matrices = np.asarray(matrices, dtype=float)
    weights = normalized_weights(len(matrices), weights)
    return matrix_exp(sum(w*matrix_log(matrix) for w, matrix in zip(weights, matrices)))


def frechet_mean(matrices, weights=None, tolerance=1e-9, max_iterations=150):
    """Karcher mean via geodesic gradient descent with Armijo backtracking.

    The objective is one half the weighted squared affine-invariant distance.
    The returned residual is the norm of the gradient in whitened coordinates.
    """
    matrices = np.asarray(matrices, dtype=float)
    weights = normalized_weights(len(matrices), weights)
    for matrix in matrices:
        power(matrix, 0)
    estimate = sum(w*matrix for w, matrix in zip(weights, matrices))

    def objective(point):
        return float(.5*sum(w*distance(point, matrix)**2 for w, matrix in zip(weights, matrices)))

    history = [objective(estimate)]
    for iteration in range(max_iterations):
        logs = [matrix_log(whiten(estimate, matrix)) for matrix in matrices]
        gradient = sum(w*value for w, value in zip(weights, logs))
        residual = float(np.linalg.norm(gradient, 'fro'))
        if residual <= tolerance:
            return estimate, {'iterations': iteration, 'gradient_norm': residual, 'objective': history[-1],
                              'objective_history': history, 'converged': True}
        root = power(estimate, .5)
        step = 1.
        for _ in range(35):
            trial = root @ matrix_exp(step*gradient) @ root
            trial = (trial+trial.T)/2
            value = objective(trial)
            if value <= history[-1]-1e-4*step*residual**2+1e-14:
                break
            step *= .5
        else:
            raise RuntimeError('Karcher mean line search failed.')
        estimate = trial
        history.append(value)
    raise RuntimeError(f'Karcher mean did not converge after {max_iterations} iterations.')
