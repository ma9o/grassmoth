"""Independent marginal StudentT integration for learned reading coefficients."""

import numpy as np
from scipy.stats import t


def student_features(joint):
    """Mixed moments and smooth joint distribution probes for five coordinates."""
    projections = np.array(
        [[1.0, 0.0, 0.0, 1.0, -1.0], [0.0, 1.0, -1.0, 0.5, 0.0], [1.0, -1.0, 0.3, 0.0, 0.4]]
    )
    return np.concatenate(
        (
            joint,
            (joint[:, :, None] * joint[:, None, :]).reshape(len(joint), -1),
            np.cos(joint @ projections.T),
            np.sin(joint @ projections.T),
        ),
        axis=1,
    )


def student_quadrature(parameter_order=16, path_order=64):
    """Stream tensor Hermite integration, using finer nodes along the observed path.

    SciPy evaluates the marginal StudentT laws; no augmented or particle densities
    enter this reference. Only quadrature weights, rather than the entire tensor
    of paths or features, are retained for independent posterior starting draws.
    """
    parameter_nodes, parameter_weights = np.polynomial.hermite.hermgauss(parameter_order)
    path_nodes, path_weights = np.polynomial.hermite.hermgauss(path_order)
    shape = (parameter_order,) * 3 + (path_order,) * 2
    count = int(np.prod(shape))
    probabilities = np.empty(count)
    first = np.zeros(36)
    second = np.zeros(36)

    def coordinates(indices):
        joint = np.sqrt(2) * np.column_stack(
            (parameter_nodes[indices[:, :3]], path_nodes[indices[:, 3:]])
        )
        joint[:, 4] = 0.7 * joint[:, 3] + np.sqrt(0.3) * joint[:, 4]
        return joint

    for start in range(0, count, 65536):
        end = min(count, start + 65536)
        indices = np.stack(np.unravel_index(np.arange(start, end), shape), axis=-1)
        joint = coordinates(indices)
        weights = (
            np.prod(parameter_weights[indices[:, :3]], axis=1)
            * np.prod(path_weights[indices[:, 3:]], axis=1)
            / np.pi**2.5
        )
        df = 3 * np.exp(0.7 * joint[:, 0])
        scale = np.exp(-0.3 + 0.3 * joint[:, 1])
        for time, readings in enumerate([[1.4, -0.4], [-0.8, 0.6]]):
            for channel, loading in enumerate([1.0, -0.7]):
                mean = joint[:, time + 3] * loading * (1 + 0.3 * joint[:, 2]) + [0.2, -0.1][channel]
                weights *= t.pdf(readings[channel], df, loc=mean, scale=scale)
        probabilities[start:end] = weights
        features = student_features(joint)
        first += weights @ features
        second += weights @ features**2
    mass = probabilities.sum()
    probabilities /= mass
    expected = first / mass
    selected = np.random.default_rng(819).choice(count, 8192, p=probabilities)
    starts = coordinates(np.stack(np.unravel_index(selected, shape), axis=-1))
    return starts, expected, second / mass - expected**2
