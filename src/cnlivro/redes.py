"""Redes neurais pequenas em PyTorch para ajuste de curvas (cap. 9).

Implementações de REFERÊNCIA, no espírito do resto do pacote: uma rede
totalmente conectada (MLP), um padronizador de variáveis e um laço de treino
que guarda o histórico da perda, para que o treino possa ser verificado como
qualquer outro método iterativo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def mlp(entradas: int = 1, saidas: int = 1, largura: int = 32, camadas: int = 3,
        ativacao: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Rede totalmente conectada com ``camadas`` camadas ocultas de ``largura`` neurônios."""
    modulos: list[nn.Module] = []
    n = entradas
    for _ in range(camadas):
        modulos += [nn.Linear(n, largura), ativacao()]
        n = largura
    modulos.append(nn.Linear(n, saidas))
    return nn.Sequential(*modulos)


class Padronizador:
    """Leva cada coluna para média 0 e desvio padrão 1, e desfaz a transformação."""

    def __init__(self, dados):
        dados = torch.as_tensor(dados, dtype=torch.get_default_dtype())
        if dados.ndim == 1:
            dados = dados[:, None]
        self.media = dados.mean(dim=0)
        self.desvio = dados.std(dim=0)
        self.desvio[self.desvio == 0] = 1.0

    def __call__(self, dados):
        dados = torch.as_tensor(dados, dtype=torch.get_default_dtype())
        if dados.ndim == 1:
            dados = dados[:, None]
        return (dados - self.media) / self.desvio

    def inverso(self, dados):
        return dados * self.desvio + self.media


@dataclass
class ResultadoTreino:
    modelo: nn.Module
    epocas: int
    historico: list[float] = field(default_factory=list)            # perda de treino por época
    historico_validacao: list[float] = field(default_factory=list)  # perda de validação por época
    instantaneos: dict = field(default_factory=dict)                # {época: saída do callback}
    motivo: str = ""

    def __repr__(self) -> str:
        val = f", validação={self.historico_validacao[-1]:.3e}" if self.historico_validacao else ""
        return f"ResultadoTreino(épocas={self.epocas}, perda={self.historico[-1]:.3e}{val})"


def treinar(modelo: nn.Module, x, y, epocas: int = 2000, lr: float = 1e-2, x_val=None, y_val=None,
            peso_decaimento: float = 0.0, instantes=(), callback=None) -> ResultadoTreino:
    """Treina ``modelo`` por mínimos quadrados (erro quadrático médio) com Adam, lote inteiro.

    ``instantes`` é uma coleção de épocas em que ``callback(modelo)`` é
    chamado e o resultado guardado em ``instantaneos`` (por exemplo, a curva
    ajustada naquele momento, para desenhar a evolução do treino).
    """
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=peso_decaimento)
    perda_fn = nn.MSELoss()
    resultado = ResultadoTreino(modelo=modelo, epocas=epocas)
    instantes = set(instantes)
    for epoca in range(1, epocas + 1):
        otimizador.zero_grad()
        perda = perda_fn(modelo(x), y)
        perda.backward()
        otimizador.step()
        resultado.historico.append(perda.item())
        if x_val is not None:
            with torch.no_grad():
                resultado.historico_validacao.append(float(perda_fn(modelo(torch.as_tensor(x_val, dtype=x.dtype)),
                                                                    torch.as_tensor(y_val, dtype=y.dtype))))
        if epoca in instantes and callback is not None:
            with torch.no_grad():
                resultado.instantaneos[epoca] = callback(modelo)
        if not np.isfinite(resultado.historico[-1]):
            resultado.motivo = "a perda virou NaN ou infinito"
            resultado.epocas = epoca
            break
    return resultado
