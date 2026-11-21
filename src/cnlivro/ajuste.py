"""Ajuste de modelos a dados por mínimos quadrados (cap. 7).

Implementações de REFERÊNCIA. Para uso real: ``np.linalg.lstsq``,
``np.polynomial.Polynomial.fit``, ``scipy.optimize.least_squares`` e
``scipy.optimize.curve_fit``.
"""

from __future__ import annotations

import numpy as np

from .lineares import ResultadoVetorial, substituicao_regressiva


def equacoes_normais(A, b):
    """Resolve min ‖Ax − b‖ por AᵀA x = Aᵀb. Simples, mas eleva κ ao quadrado."""
    A = np.asarray(A, dtype=float)
    return np.linalg.solve(A.T @ A, A.T @ np.asarray(b, dtype=float))


def minimos_quadrados_qr(A, b):
    """Resolve min ‖Ax − b‖ por QR: A = QR, depois R x = Qᵀb. Não eleva κ ao quadrado."""
    Q, R = np.linalg.qr(np.asarray(A, dtype=float))
    return substituicao_regressiva(R, Q.T @ np.asarray(b, dtype=float))


def r2(y, y_ajustado):
    """Coeficiente de determinação R² = 1 − SQres/SQtot."""
    y = np.asarray(y, dtype=float)
    return 1.0 - np.sum((y - y_ajustado) ** 2) / np.sum((y - y.mean()) ** 2)


def gauss_newton(residuo, jacobiana, p0, tol=1e-10, max_iter=100):
    """Gauss-Newton para min ½‖r(p)‖²: cada passo resolve J Δp ≈ −r por QR.

    ``erros`` guarda ‖Δp‖/‖p‖. Sem busca linear: pode divergir longe da solução
    (o Levenberg-Marquardt de ``scipy.optimize.least_squares`` é a versão robusta).
    """
    p = np.array(p0, dtype=float)
    historico, erros = [p.copy()], []
    for k in range(1, max_iter + 1):
        dp = minimos_quadrados_qr(jacobiana(p), -residuo(p))
        p = p + dp
        historico.append(p.copy())
        erros.append(float(np.linalg.norm(dp) / (np.linalg.norm(p) or 1.0)))
        if not np.all(np.isfinite(p)):
            return ResultadoVetorial(p, False, k, historico, erros, "divergiu")
        if erros[-1] < tol:
            return ResultadoVetorial(p, True, k, historico, erros, "‖Δp‖/‖p‖ < tol")
    return ResultadoVetorial(p, False, max_iter, historico, erros, "max_iter atingido")
