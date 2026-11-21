"""Ferramentas para VERIFICAR código numérico.

É o módulo mais importante para o aluno que trabalha com um agente de IA: o
agente escreve o método, e estas funções ajudam a provar que ele se comporta
como a teoria diz.
"""

from __future__ import annotations

import numpy as np


def ordem_observada(h, erro) -> float:
    """Ordem de convergência p estimada a partir de erro ≈ C·h^p.

    Ajusta uma reta a log(erro) × log(h) por mínimos quadrados e devolve a
    inclinação. Pontos com erro nulo (ou dominados por arredondamento) devem
    ser removidos antes.
    """
    h = np.asarray(h, dtype=float)
    erro = np.asarray(erro, dtype=float)
    mascara = (h > 0) & (erro > 0)
    if mascara.sum() < 2:
        raise ValueError("São necessários pelo menos dois pontos com erro positivo.")
    p, _ = np.polyfit(np.log(h[mascara]), np.log(erro[mascara]), 1)
    return float(p)


def ordem_iterativa(historico, raiz: float, descartar_final: int = 1) -> float:
    """Ordem q de um método iterativo, a partir de |e_{k+1}| ≈ C·|e_k|^q.

    Usa os erros verdadeiros e_k = x_k − raiz. Os últimos passos costumam
    estar dominados pelo arredondamento; ``descartar_final`` os ignora.
    """
    e = np.abs(np.asarray(historico, dtype=float) - raiz)
    if descartar_final:
        e = e[:-descartar_final]
    e = e[e > 0]
    if e.size < 3:
        raise ValueError("Histórico curto demais para estimar a ordem.")
    q, _ = np.polyfit(np.log(e[:-1]), np.log(e[1:]), 1)
    return float(q)


def compara(valor, referencia, rtol: float = 1e-8, atol: float = 0.0) -> bool:
    """Verifica se ``valor`` concorda com ``referencia`` e explica a diferença."""
    valor = np.asarray(valor, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    ok = bool(np.allclose(valor, referencia, rtol=rtol, atol=atol))
    dif = float(np.max(np.abs(valor - referencia)))
    print(f"{'OK' if ok else 'FALHOU'}: diferença máxima = {dif:.3e} (rtol = {rtol:g})")
    return ok
