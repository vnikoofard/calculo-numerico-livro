"""Interpolação polinomial e por splines (cap. 6).

Implementações de REFERÊNCIA, curtas e legíveis. Para uso real, prefira
``scipy.interpolate`` (``BarycentricInterpolator``, ``CubicSpline``,
``PchipInterpolator``).
"""

from __future__ import annotations

import numpy as np


def diferencas_divididas(x, y):
    """Coeficientes da forma de Newton: f[x0], f[x0,x1], ..., f[x0,...,xn]."""
    x = np.asarray(x, dtype=float)
    c = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j - 1:-1]) / (x[j:] - x[:n - j])
    return c


def newton_avalia(coef, x_nos, t):
    """Avalia o polinômio de Newton em t (esquema de Horner)."""
    t = np.asarray(t, dtype=float)
    p = np.full_like(t, coef[-1])
    for k in range(len(coef) - 2, -1, -1):
        p = p * (t - x_nos[k]) + coef[k]
    return p


def lagrange(x_nos, y_nos, t):
    """Avalia o polinômio interpolador pela fórmula baricêntrica (estável, O(n) por ponto)."""
    x_nos = np.asarray(x_nos, dtype=float)
    y_nos = np.asarray(y_nos, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(x_nos)
    w = np.ones(n)
    for j in range(n):
        w[j] = 1.0 / np.prod(x_nos[j] - np.delete(x_nos, j))
    dif = t[:, None] - x_nos[None, :]
    exato = dif == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        termos = w / dif
        p = (termos @ y_nos) / termos.sum(axis=1)
    linhas, colunas = np.nonzero(exato)
    p[linhas] = y_nos[colunas]
    return p


def nos_chebyshev(n, a=-1.0, b=1.0):
    """n nós de Chebyshev (zeros de T_n) no intervalo [a, b], em ordem crescente."""
    k = np.arange(n)
    x = np.cos((2 * k + 1) * np.pi / (2 * n))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def spline_cubica_natural(x, y):
    """Spline cúbica natural (S'' = 0 nas pontas). Devolve uma função S(t).

    As segundas derivadas M_i nos nós saem de um sistema TRIDIAGONAL (cap. 4).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Os nós precisam estar em ordem estritamente crescente.")
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    r = np.zeros(n + 1)
    A[0, 0] = A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, r)

    def S(t):
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(x, t) - 1, 0, n - 1)
        dx0, dx1, hi = t - x[i], x[i + 1] - t, h[i]
        return (M[i] * dx1**3 + M[i + 1] * dx0**3) / (6 * hi) \
            + (y[i] / hi - M[i] * hi / 6) * dx1 + (y[i + 1] / hi - M[i + 1] * hi / 6) * dx0

    return S
