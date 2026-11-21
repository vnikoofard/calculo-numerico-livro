"""Integração numérica (cap. 11).

Implementações de REFERÊNCIA: regras de Newton-Cotes compostas, Romberg,
quadratura de Gauss-Legendre, Simpson adaptativo e Monte Carlo. Os métodos
iterativos ou adaptativos devolvem :class:`ResultadoIntegral` com o histórico.
Para uso em produção, ``scipy.integrate.quad`` é a escolha padrão.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ResultadoIntegral:
    valor: float
    convergiu: bool
    avaliacoes: int                                   # número de avaliações de f
    historico: list[float] = field(default_factory=list)
    erros: list[float] = field(default_factory=list)  # erro estimado em cada etapa
    motivo: str = ""
    intervalos: list[tuple[float, float]] = field(default_factory=list)
    tabela: np.ndarray | None = None

    def __repr__(self) -> str:
        estado = "convergiu" if self.convergiu else f"NÃO convergiu ({self.motivo})"
        return f"ResultadoIntegral(valor={self.valor:.15g}, {estado}, avaliações={self.avaliacoes})"


def trapezio(f: Callable, a: float, b: float, n: int) -> float:
    """Regra do trapézio composta com n subintervalos iguais (f vetorizada)."""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))


def simpson(f: Callable, a: float, b: float, n: int) -> float:
    """Regra 1/3 de Simpson composta; n (número de subintervalos) deve ser par."""
    if n % 2:
        raise ValueError("A regra 1/3 de Simpson exige um número PAR de subintervalos.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1]))


def trapezio_dados(x, y) -> float:
    """Trapézio para dados com espaçamento qualquer (o mesmo que ``np.trapezoid``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def gauss_legendre(f: Callable, a: float, b: float, n: int, subintervalos: int = 1) -> float:
    """Quadratura de Gauss-Legendre com n pontos em cada subintervalo.

    É exata para polinômios de grau até 2n − 1.
    """
    t, w = np.polynomial.legendre.leggauss(n)
    bordas = np.linspace(a, b, subintervalos + 1)
    total = 0.0
    for c, d in zip(bordas[:-1], bordas[1:]):
        meio, raio = (c + d) / 2, (d - c) / 2
        total += raio * np.sum(w * f(meio + raio * t))
    return float(total)


def romberg(f: Callable, a: float, b: float, tol: float = 1e-10, max_niveis: int = 20) -> ResultadoIntegral:
    """Integração de Romberg: trapézios com h, h/2, h/4... e extrapolação de Richardson.

    Reaproveita as avaliações do nível anterior. Para quando duas diagonais
    sucessivas da tabela concordam dentro de ``tol`` (relativa).
    """
    R = np.full((max_niveis, max_niveis), np.nan)
    h = b - a
    fa, fb = f(np.array([a, b]))
    R[0, 0] = h * (fa + fb) / 2
    avaliacoes = 2
    historico, erros = [float(R[0, 0])], []
    for i in range(1, max_niveis):
        h /= 2
        novos = a + h * np.arange(1, 2**i, 2)
        R[i, 0] = R[i - 1, 0] / 2 + h * np.sum(f(novos))
        avaliacoes += novos.size
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
        historico.append(float(R[i, i]))
        erro = abs(R[i, i] - R[i - 1, i - 1])
        erros.append(float(erro))
        if erro <= tol * max(abs(R[i, i]), 1e-300):
            return ResultadoIntegral(float(R[i, i]), True, avaliacoes, historico, erros,
                                     tabela=R[: i + 1, : i + 1])
    return ResultadoIntegral(float(R[-1, -1]), False, avaliacoes, historico, erros,
                             motivo="número máximo de níveis", tabela=R)


def simpson_adaptativo(f: Callable, a: float, b: float, tol: float = 1e-8,
                       max_profundidade: int = 50) -> ResultadoIntegral:
    """Simpson adaptativo: divide ao meio só os intervalos onde o erro estimado é grande.

    O erro de S(c, d) é estimado por |S(c, m) + S(m, d) − S(c, d)| / 15, e a
    tolerância é repartida entre as metades. Devolve os intervalos usados.
    ``f`` é chamada com escalares.
    """
    cache: dict[float, float] = {}

    def fx(x):
        if x not in cache:
            cache[x] = float(f(x))
        return cache[x]

    def s(c, d):
        m = (c + d) / 2
        return (d - c) / 6 * (fx(c) + 4 * fx(m) + fx(d))

    intervalos = []
    estado = {"ok": True}

    def recursao(c, d, inteiro, tol_local, prof):
        m = (c + d) / 2
        esq, dir_ = s(c, m), s(m, d)
        diferenca = esq + dir_ - inteiro
        if abs(diferenca) <= 15 * tol_local or prof >= max_profundidade:
            if prof >= max_profundidade:
                estado["ok"] = False
            intervalos.append((c, d))
            return esq + dir_ + diferenca / 15
        return (recursao(c, m, esq, tol_local / 2, prof + 1)
                + recursao(m, d, dir_, tol_local / 2, prof + 1))

    valor = recursao(a, b, s(a, b), tol, 0)
    intervalos.sort()
    return ResultadoIntegral(float(valor), estado["ok"], len(cache), [float(valor)], [],
                             motivo="" if estado["ok"] else "profundidade máxima atingida",
                             intervalos=intervalos)


def monte_carlo(f: Callable, limites, n: int, rng=None) -> ResultadoIntegral:
    """Integral de f numa caixa por Monte Carlo, com o erro padrão da estimativa.

    ``limites`` é uma lista de pares (a, b), um por dimensão; ``f`` recebe uma
    matriz n × d e devolve n valores. O erro padrão é s·V/√n, com s o desvio
    padrão amostral dos valores de f e V o volume da caixa.
    """
    rng = np.random.default_rng() if rng is None else rng
    limites = np.asarray(limites, dtype=float).reshape(-1, 2)
    a, b = limites[:, 0], limites[:, 1]
    volume = float(np.prod(b - a))
    pontos = a + (b - a) * rng.random((n, limites.shape[0]))
    valores = np.asarray(f(pontos), dtype=float)
    media = valores.mean()
    erro_padrao = volume * valores.std(ddof=1) / np.sqrt(n)
    return ResultadoIntegral(float(volume * media), True, n, [float(volume * media)], [float(erro_padrao)])
