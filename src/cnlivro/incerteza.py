"""Propagação de incerteza e sensibilidade (apêndice A).

Implementações de REFERÊNCIA, curtas e legíveis:

- Monte Carlo com erro padrão (:func:`monte_carlo`);
- probabilidade de um evento com intervalo de confiança de Wilson (:func:`probabilidade`);
- propagação linear (método delta) com derivadas por diferenças centradas (:func:`propagacao_linear`);
- gradiente por diferenciação automática com PyTorch (:func:`gradiente_torch`).

Para uso em produção: ``scipy.stats`` (distribuições, ``qmc`` para hipercubo latino e
sequências de Sobol) e bibliotecas de análise de sensibilidade como SALib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm


@dataclass
class ResultadoMC:
    media: float
    desvio: float                     # desvio-padrão da saída
    erro_padrao: float                # erro padrão da média, desvio/√N
    amostras: np.ndarray              # saídas do modelo, uma por amostra
    n: int
    historico: list = field(default_factory=list)   # média acumulada a cada potência de 10

    def __repr__(self) -> str:
        return f"ResultadoMC(média={self.media:.6g} ± {self.erro_padrao:.2g}, desvio={self.desvio:.4g}, n={self.n})"


def monte_carlo(modelo: Callable, entradas: dict, n: int | None = None) -> ResultadoMC:
    """Avalia ``modelo(**entradas)`` em todas as amostras de uma vez e resume a saída.

    ``entradas`` mapeia o nome de cada parâmetro para um vetor de amostras (todos do mesmo
    tamanho). O modelo deve ser vetorizado. Saídas não finitas (``nan``, ``inf``) são
    contadas na média: se o modelo as produz, trate-as antes (por exemplo, como um evento).
    """
    saida = np.asarray(modelo(**entradas), dtype=float)
    n = saida.size if n is None else n
    media_acum = [float(np.mean(saida[:m])) for m in (10 ** np.arange(1, int(np.log10(n)) + 1)) if m <= n]
    desvio = float(np.std(saida, ddof=1))
    return ResultadoMC(float(np.mean(saida)), desvio, desvio / np.sqrt(n), saida, n, media_acum)


def probabilidade(eventos, confianca: float = 0.95) -> tuple[float, float, float]:
    """Estima P(evento) a partir de um vetor booleano, com o intervalo de Wilson.

    Devolve (p, limite inferior, limite superior). Ao contrário do intervalo
    p ± z√(p(1-p)/N), o de Wilson não colapsa em zero quando nenhum evento é observado.
    """
    eventos = np.asarray(eventos, dtype=bool)
    N, k = eventos.size, int(eventos.sum())
    p = k / N
    z = norm.ppf(0.5 + confianca / 2)
    centro = (p + z**2 / (2 * N)) / (1 + z**2 / N)
    meia = z / (1 + z**2 / N) * np.sqrt(p * (1 - p) / N + z**2 / (4 * N**2))
    return p, max(0.0, centro - meia), min(1.0, centro + meia)


def propagacao_linear(f: Callable, x0, sigmas, passo_rel: float = 1e-6):
    """Método delta: σ_y² ≈ Σ (∂f/∂x_i)² σ_i², para entradas independentes.

    As derivadas são calculadas por diferenças centradas com passo relativo ``passo_rel``
    (n entradas custam 2n avaliações de f). Devolve (f(x0), σ_y, gradiente).
    """
    x0 = np.asarray(x0, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    y0 = float(f(*x0))
    grad = np.zeros_like(x0)
    for i in range(x0.size):
        h = passo_rel * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (f(*xp) - f(*xm)) / (2 * h)
    return y0, float(np.sqrt(np.sum((grad * sigmas) ** 2))), grad


def gradiente_torch(f: Callable, x0):
    """Valor e gradiente de f (escrita com operações do PyTorch) em x0, por diferenciação automática.

    Um único passo para trás (modo reverso) dá as derivadas em relação a todas as entradas.
    Devolve (f(x0), gradiente) como números/arrays do NumPy, em precisão dupla.
    """
    import torch

    x = torch.tensor(np.asarray(x0, dtype=float), dtype=torch.float64, requires_grad=True)
    y = f(*x)
    (g,) = torch.autograd.grad(y, x)
    return float(y.detach()), g.numpy()
