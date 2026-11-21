"""Derivação numérica (cap. 10).

Implementações de REFERÊNCIA: curtas e legíveis, para comparar com o código
pedido ao agente. Para derivar dados medidos com espaçamento qualquer,
``derivada_dados`` faz o mesmo que ``np.gradient(y, x, edge_order=2)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Funcao = Callable[[float], float]


def progressiva(f: Funcao, x: float, h: float) -> float:
    """Diferença progressiva, erro O(h)."""
    return (f(x + h) - f(x)) / h


def regressiva(f: Funcao, x: float, h: float) -> float:
    """Diferença regressiva, erro O(h)."""
    return (f(x) - f(x - h)) / h


def centrada(f: Funcao, x: float, h: float) -> float:
    """Diferença centrada, erro O(h²)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def segunda_centrada(f: Funcao, x: float, h: float) -> float:
    """Segunda derivada por diferença centrada, erro O(h²)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def passo_complexo(f: Funcao, x: float, h: float = 1e-20) -> float:
    """Derivada pelo passo complexo: Im f(x + ih) / h.

    Não há subtração de números próximos, então h pode ser minúsculo. Exige
    que ``f`` aceite números complexos e seja analítica (sem ``abs``, ``max``...).
    """
    return float(np.imag(f(x + 1j * h)) / h)


@dataclass
class TabelaExtrapolacao:
    """Resultado da extrapolação de Richardson (ou de Romberg)."""

    valor: float
    tabela: np.ndarray                 # tabela[i, j]: passo h/2^i, j extrapolações
    passos: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"TabelaExtrapolacao(valor={self.valor:.15g}, níveis={self.tabela.shape[0]})"


def richardson(f: Funcao, x: float, h: float, niveis: int = 4, metodo: Callable = centrada,
               ordem: int = 2, salto: int = 2) -> TabelaExtrapolacao:
    """Extrapolação de Richardson de uma fórmula de diferenças.

    A fórmula ``metodo`` tem erro c₁hᵖ + c₂hᵖ⁺ˢ + ..., com p = ``ordem`` e
    s = ``salto`` (para a diferença centrada, p = 2 e s = 2, pois só aparecem
    potências pares). Cada coluna da tabela elimina um termo do erro.
    """
    passos = [h / 2**i for i in range(niveis)]
    T = np.full((niveis, niveis), np.nan)
    for i, hi in enumerate(passos):
        T[i, 0] = metodo(f, x, hi)
        for j in range(1, i + 1):
            fator = 2.0 ** (ordem + salto * (j - 1))
            T[i, j] = T[i, j - 1] + (T[i, j - 1] - T[i - 1, j - 1]) / (fator - 1)
    return TabelaExtrapolacao(valor=float(T[-1, -1]), tabela=T, passos=passos)


def derivada_dados(x, y) -> np.ndarray:
    """Primeira derivada de dados (x, y) com espaçamento qualquer, erro O(h²).

    No interior usa a parábola pelos três pontos vizinhos; nas pontas, a
    fórmula unilateral de três pontos. Os ``x`` devem ser crescentes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("São necessários pelo menos três pontos.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Os valores de x devem ser estritamente crescentes.")
    d = np.empty_like(y)
    h0, h1 = h[:-1], h[1:]                      # h0 = x_i − x_{i−1}, h1 = x_{i+1} − x_i
    d[1:-1] = (-h1 / (h0 * (h0 + h1)) * y[:-2] + (h1 - h0) / (h0 * h1) * y[1:-1]
               + h0 / (h1 * (h0 + h1)) * y[2:])
    a, b = h[0], h[1]
    d[0] = -(2 * a + b) / (a * (a + b)) * y[0] + (a + b) / (a * b) * y[1] - a / (b * (a + b)) * y[2]
    a, b = h[-1], h[-2]
    d[-1] = (2 * a + b) / (a * (a + b)) * y[-1] - (a + b) / (a * b) * y[-2] + a / (b * (a + b)) * y[-3]
    return d
