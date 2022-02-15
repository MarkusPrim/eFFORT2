import functools
from scipy.special import roots_legendre
import numpy as np


def quad(f, a, b, order=19, **kwargs):
    """Gaussiand Quadrature with Legendre polynomials. Same signature and return value as scipy.integrate.quad."""
    cached_roots_legendre = functools.lru_cache(roots_legendre)
    roots, weights = cached_roots_legendre(order)
    return (b - a) / 2 * sum(w_i * f((b - a) / 2 * x_i + (b + a) / 2) for w_i, x_i in zip(weights, roots)), np.nan
