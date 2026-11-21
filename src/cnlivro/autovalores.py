"""Autovalores e autovetores (cap. 8).

Implementações de REFERÊNCIA do método da potência e da potência inversa.
Para uso real: ``np.linalg.eigh``/``scipy.linalg.eigh`` (simétricas, inclusive
o problema generalizado K v = λ M v) e ``scipy.sparse.linalg.eigsh``.
"""

from __future__ import annotations

import numpy as np

from .lineares import ResultadoVetorial


def metodo_potencia(A, x0=None, tol=1e-10, max_iter=10_000, deslocamento=0.0, inversa=False):
    """Método da potência (ou da potência inversa com deslocamento σ).

    - ``inversa=False``: converge para o autovalor de MAIOR módulo de A.
    - ``inversa=True``: aplica (A − σI)⁻¹ e converge para o autovalor de A mais
      PRÓXIMO de σ = ``deslocamento``.

    O autovalor é estimado pelo quociente de Rayleigh. ``x`` do resultado é o
    autovetor (norma 1); o autovalor fica em ``historico_autovalor[-1]``, e
    ``erros`` guarda a variação relativa do autovalor a cada passo.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n) if x0 is None else np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    if inversa:
        from scipy.linalg import lu_factor, lu_solve
        fatores = lu_factor(A - deslocamento * np.eye(n))
        aplica = lambda v: lu_solve(fatores, v)
    else:
        aplica = lambda v: A @ v
    lam = x @ A @ x
    historico, autovalores, erros = [x.copy()], [lam], []
    for k in range(1, max_iter + 1):
        y = aplica(x)
        x = y / np.linalg.norm(y)
        lam_novo = x @ A @ x                  # quociente de Rayleigh
        erros.append(abs(lam_novo - lam) / (abs(lam_novo) or 1.0))
        lam = lam_novo
        historico.append(x.copy())
        autovalores.append(lam)
        if erros[-1] < tol:
            r = ResultadoVetorial(x, True, k, historico, erros, "variação relativa de λ < tol")
            r.historico_autovalor = autovalores
            return r
    r = ResultadoVetorial(x, False, max_iter, historico, erros, "max_iter atingido")
    r.historico_autovalor = autovalores
    return r


def predio_cisalhamento(massas, rigidezes):
    """Matrizes de massa M e rigidez K de um prédio de cisalhamento.

    O andar i tem massa ``massas[i]``; a coluna entre o andar i−1 (ou o chão)
    e o andar i tem rigidez ``rigidezes[i]``. Frequências: K v = ω² M v.
    """
    m = np.asarray(massas, dtype=float)
    k = np.asarray(rigidezes, dtype=float)
    n = len(m)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i + 1 < n:
            K[i, i] += k[i + 1]
            K[i, i + 1] = K[i + 1, i] = -k[i + 1]
    return np.diag(m), K
