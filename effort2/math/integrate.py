from scipy.special import roots_legendre, eval_legendre


def quad(f, a, b, order=19):
    roots, weights = roots_legendre(order)
    return (b - a) / 2 * sum(w_i * f((b - a) / 2 * x_i + (b + a) / 2) for w_i, x_i in zip(weights, roots))