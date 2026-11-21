"""Desempenho: medir tempo direito e resolver muitos sistemas de uma vez (apêndice B).

- :func:`cronometrar`: mediana de várias execuções, com aquecimento (compilação JIT,
  caches) e sincronização opcional da GPU;
- :func:`dispositivo`: "cuda", "mps" ou "cpu", conforme o que o PyTorch encontrar;
- :func:`thomas` e :func:`thomas_lote`: algoritmo de Thomas para um sistema tridiagonal,
  ou para um lote de sistemas independentes, vetorizado sobre o lote.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ResultadoTempo:
    mediana: float                    # s
    minimo: float                     # s
    tempos: list = field(default_factory=list)
    valor: object = None              # o que a função devolveu na última execução

    def __repr__(self) -> str:
        return f"ResultadoTempo(mediana={self.mediana:.3g} s, mínimo={self.minimo:.3g} s, execuções={len(self.tempos)})"


def cronometrar(f: Callable, *args, repeticoes: int = 5, aquecimento: int = 1, sincronizar: bool = False,
                **kwargs) -> ResultadoTempo:
    """Mede o tempo de ``f(*args, **kwargs)``.

    As ``aquecimento`` primeiras execuções não contam (compilação JIT, alocação, caches).
    Com ``sincronizar=True``, espera a GPU terminar antes de parar o relógio: as operações
    do PyTorch na GPU são assíncronas, e sem isso mede-se só o tempo de enfileirá-las.
    """
    def sinc():
        if sincronizar:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    valor = None
    for _ in range(aquecimento):
        valor = f(*args, **kwargs)
    sinc()
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        valor = f(*args, **kwargs)
        sinc()
        tempos.append(time.perf_counter() - t0)
    return ResultadoTempo(float(np.median(tempos)), float(np.min(tempos)), tempos, valor)


def dispositivo() -> str:
    """O melhor dispositivo disponível para o PyTorch: "cuda", "mps" ou "cpu"."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def thomas(a, b, c, d) -> np.ndarray:
    """Resolve o sistema tridiagonal com subdiagonal a, diagonal b, superdiagonal c e lado direito d.

    Todos com o mesmo comprimento n (a[0] e c[-1] são ignorados). Custo O(n), sem pivotamento:
    adequado para matrizes diagonalmente dominantes, como as das diferenças finitas.
    """
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    return thomas_lote(a[None], b[None], c[None], d[None])[0]


def thomas_lote(a, b, c, d) -> np.ndarray:
    """Algoritmo de Thomas para um lote de sistemas tridiagonais independentes.

    Cada argumento tem forma (lote, n); a linha k descreve o k-ésimo sistema. O laço é
    sobre as n incógnitas, e cada operação age sobre o lote inteiro de uma vez.

    Internamente, os arranjos são copiados na forma (n, lote): assim, a "coluna" i de todos
    os sistemas fica contígua na memória, o que é várias vezes mais rápido para lotes grandes
    do que percorrer colunas de um arranjo (lote, n), com saltos de n números.
    """
    a, b, c, d = (np.ascontiguousarray(np.asarray(v, dtype=float).T) for v in (a, b, c, d))
    n = b.shape[0]
    cp, dp = np.empty_like(b), np.empty_like(b)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty_like(b)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.ascontiguousarray(x.T)
