"""Métodos para encontrar raízes de equações f(x) = 0.

Estas são as implementações de REFERÊNCIA do livro. Elas são curtas e
legíveis de propósito: servem para comparar com o código que o aluno pede ao
agente, não para substituir ``scipy.optimize``.

Todas as funções têm a mesma interface de saída: devolvem um objeto
:class:`Resultado` com a raiz e o histórico completo das iterações. Esse
histórico é o que permite verificar o método (medir a ordem de convergência,
desenhar as iterações, ver onde ele falha).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Funcao = Callable[[float], float]


@dataclass
class Resultado:
    """Saída comum a todos os métodos de raízes."""

    raiz: float
    convergiu: bool
    iteracoes: int
    historico: list[float] = field(default_factory=list)  # x_0, x_1, ..., x_n
    erros: list[float] = field(default_factory=list)      # erro relativo aproximado em cada passo
    motivo: str = ""

    def __repr__(self) -> str:
        estado = "convergiu" if self.convergiu else f"NÃO convergiu ({self.motivo})"
        return f"Resultado(raiz={self.raiz:.12g}, {estado}, iterações={self.iteracoes})"


def _erro_relativo(novo: float, velho: float) -> float:
    if novo == 0.0:
        return abs(novo - velho)
    return abs((novo - velho) / novo)


def bissecao(f: Funcao, a: float, b: float, tol: float = 1e-10, max_iter: int = 100) -> Resultado:
    """Método da bisseção no intervalo [a, b], com f(a)·f(b) < 0."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) precisam ter sinais opostos.")
    historico, erros = [], []
    x_velho = a
    for k in range(1, max_iter + 1):
        x = (a + b) / 2
        fx = f(x)
        historico.append(x)
        erros.append(_erro_relativo(x, x_velho))
        if fx == 0 or erros[-1] < tol:
            return Resultado(x, True, k, historico, erros)
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_velho = x
    return Resultado(x, False, max_iter, historico, erros, "máximo de iterações")


def falsa_posicao(f: Funcao, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> Resultado:
    """Método da falsa posição (regula falsi) no intervalo [a, b]."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) precisam ter sinais opostos.")
    historico, erros = [], []
    x_velho = a
    for k in range(1, max_iter + 1):
        x = b - fb * (a - b) / (fa - fb)
        fx = f(x)
        historico.append(x)
        erros.append(_erro_relativo(x, x_velho))
        if fx == 0 or erros[-1] < tol:
            return Resultado(x, True, k, historico, erros)
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_velho = x
    return Resultado(x, False, max_iter, historico, erros, "máximo de iterações")


def ponto_fixo(g: Funcao, x0: float, tol: float = 1e-10, max_iter: int = 200) -> Resultado:
    """Iteração de ponto fixo x_{k+1} = g(x_k)."""
    historico, erros = [x0], []
    x = x0
    for k in range(1, max_iter + 1):
        try:
            x_novo = g(x)
        except (OverflowError, ValueError) as e:
            return Resultado(x, False, k, historico, erros, f"erro numérico: {e}")
        historico.append(x_novo)
        erros.append(_erro_relativo(x_novo, x))
        x = x_novo
        if erros[-1] < tol:
            return Resultado(x, True, k, historico, erros)
    return Resultado(x, False, max_iter, historico, erros, "máximo de iterações")


def newton(
    f: Funcao, df: Funcao, x0: float, tol: float = 1e-10, max_iter: int = 50, m: int = 1
) -> Resultado:
    """Método de Newton-Raphson: x_{k+1} = x_k - m·f(x_k)/f'(x_k).

    ``m`` é a multiplicidade da raiz. Com ``m = 1`` (padrão) é o método
    clássico; com a multiplicidade correta, recupera a convergência
    quadrática em raízes múltiplas.
    """
    historico, erros = [x0], []
    x = x0
    for k in range(1, max_iter + 1):
        d = df(x)
        if d == 0:
            return Resultado(x, False, k - 1, historico, erros, "derivada nula")
        x_novo = x - m * f(x) / d
        historico.append(x_novo)
        erros.append(_erro_relativo(x_novo, x))
        x = x_novo
        if erros[-1] < tol:
            return Resultado(x, True, k, historico, erros)
    return Resultado(x, False, max_iter, historico, erros, "máximo de iterações")


def secante(f: Funcao, x0: float, x1: float, tol: float = 1e-10, max_iter: int = 50) -> Resultado:
    """Método da secante, que troca f'(x) por um quociente de diferenças."""
    historico, erros = [x0, x1], []
    f0, f1 = f(x0), f(x1)
    for k in range(1, max_iter + 1):
        if f1 == f0:
            return Resultado(x1, False, k - 1, historico, erros, "divisão por zero (f(x0) = f(x1))")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        historico.append(x2)
        erros.append(_erro_relativo(x2, x1))
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
        if erros[-1] < tol:
            return Resultado(x2, True, k, historico, erros)
    return Resultado(x1, False, max_iter, historico, erros, "máximo de iterações")
