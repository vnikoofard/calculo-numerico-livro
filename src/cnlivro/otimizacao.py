"""Sistemas não lineares e otimização sem restrições (cap. 5).

Implementações de REFERÊNCIA, curtas e legíveis. Servem para comparar com o
código que o aluno pede ao agente, não para substituir ``scipy.optimize``.
Todas devolvem um :class:`~cnlivro.lineares.ResultadoVetorial` com o histórico.
"""

from __future__ import annotations

import numpy as np

from .lineares import ResultadoVetorial


def newton_sistema(F, J, x0, tol=1e-10, max_iter=50):
    """Newton para F(x) = 0, com F: Rⁿ → Rⁿ e jacobiana J(x).

    Cada passo resolve J(x_k) Δx = −F(x_k). Para quando ‖Δx‖/‖x‖ < tol.
    ``erros`` guarda ‖Δx‖/‖x‖ em cada passo.
    """
    x = np.array(x0, dtype=float)
    historico, erros = [x.copy()], []
    for k in range(1, max_iter + 1):
        try:
            dx = np.linalg.solve(np.atleast_2d(J(x)), -np.atleast_1d(F(x)))
        except np.linalg.LinAlgError:
            return ResultadoVetorial(x, False, k, historico, erros, "jacobiana singular")
        x = x + dx
        historico.append(x.copy())
        erros.append(float(np.linalg.norm(dx) / (np.linalg.norm(x) or 1.0)))
        if not np.all(np.isfinite(x)):
            return ResultadoVetorial(x, False, k, historico, erros, "divergiu")
        if erros[-1] < tol:
            return ResultadoVetorial(x, True, k, historico, erros, "‖Δx‖/‖x‖ < tol")
    return ResultadoVetorial(x, False, max_iter, historico, erros, "max_iter atingido")


def busca_armijo(f, x, fx, g, d, alfa=1.0, c=1e-4, reducao=0.5, max_reducoes=60):
    """Busca linear com retrocesso: reduz α até f(x + α d) ≤ f(x) + c α ∇f·d.

    Devolve ``None`` se nenhum α serve. Isso acontece perto do mínimo, quando a
    diminuição de f que se poderia ganhar é menor que o arredondamento de f.
    """
    inclinacao = g @ d
    for _ in range(max_reducoes):
        if f(x + alfa * d) <= fx + c * alfa * inclinacao:
            return alfa
        alfa *= reducao
    return None


def gradiente_descendente(f, grad, x0, passo=None, tol=1e-8, max_iter=10_000):
    """Descida pelo gradiente: x_{k+1} = x_k − α_k ∇f(x_k).

    ``passo=None`` usa busca linear de Armijo; um número usa passo fixo.
    Para quando ‖∇f‖ < tol. ``erros`` guarda ‖∇f(x_k)‖.
    """
    x = np.array(x0, dtype=float)
    historico, erros = [x.copy()], []
    for k in range(1, max_iter + 1):
        g = np.asarray(grad(x), dtype=float)
        erros.append(float(np.linalg.norm(g)))
        if erros[-1] < tol:
            return ResultadoVetorial(x, True, k - 1, historico, erros, "‖∇f‖ < tol")
        alfa = passo if passo is not None else busca_armijo(f, x, f(x), g, -g)
        if alfa is None:
            return ResultadoVetorial(x, False, k - 1, historico, erros,
                                     "busca linear sem progresso (limite do arredondamento de f?)")
        x = x - alfa * g
        historico.append(x.copy())
        if not np.all(np.isfinite(x)):
            return ResultadoVetorial(x, False, k, historico, erros, "divergiu")
    erros.append(float(np.linalg.norm(grad(x))))
    return ResultadoVetorial(x, erros[-1] < tol, max_iter, historico, erros, "max_iter atingido")


def bfgs(f, grad, x0, tol=1e-8, max_iter=500):
    """Quase-Newton BFGS com busca de Armijo.

    Mantém uma aproximação H da INVERSA da hessiana, atualizada com
    s = x_{k+1} − x_k e y = ∇f_{k+1} − ∇f_k. Pula a atualização se s·y ≤ 0.
    Para quando ‖∇f‖ < tol. ``erros`` guarda ‖∇f(x_k)‖.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    g = np.asarray(grad(x), dtype=float)
    historico, erros = [x.copy()], [float(np.linalg.norm(g))]
    for k in range(1, max_iter + 1):
        if erros[-1] < tol:
            return ResultadoVetorial(x, True, k - 1, historico, erros, "‖∇f‖ < tol")
        d = -H @ g
        if g @ d >= 0:          # H deixou de ser positiva definida: recomeça
            H = np.eye(n)
            d = -g
        alfa = busca_armijo(f, x, f(x), g, d)
        if alfa is None:
            return ResultadoVetorial(x, False, k - 1, historico, erros,
                                     "busca linear sem progresso (limite do arredondamento de f?)")
        x_novo = x + alfa * d
        g_novo = np.asarray(grad(x_novo), dtype=float)
        s, y = x_novo - x, g_novo - g
        sy = s @ y
        if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            rho = 1.0 / sy
            V = np.eye(n) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        x, g = x_novo, g_novo
        historico.append(x.copy())
        erros.append(float(np.linalg.norm(g)))
    return ResultadoVetorial(x, erros[-1] < tol, max_iter, historico, erros, "max_iter atingido")


def rosenbrock(x, a=1.0, b=100.0):
    """Função de Rosenbrock, o vale curvo clássico; mínimo em (a, a²)."""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x, a=1.0, b=100.0):
    return np.array([-2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
                     2 * b * (x[1] - x[0] ** 2)])
