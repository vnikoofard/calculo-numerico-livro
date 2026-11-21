"""Executa os notebooks do livro e grava as saídas neles, antes do build.

Uso: python scripts/executar_capitulos.py livro/*.ipynb

Por que não deixar o ``jupyter book build --execute`` executar? Porque ele
guarda cada pedaço da saída de texto separadamente: um ``print`` dentro de um
laço demorado vira várias caixas, e no PDF aparecem linhas em branco no meio
de uma tabela. Aqui os pedaços consecutivos de ``stdout``/``stderr`` são
juntados num só antes de gravar o notebook.

Use só no CI (ou numa cópia): os notebooks do repositório ficam sem saídas.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def juntar_fluxos(nb) -> int:
    """Junta saídas de texto consecutivas do mesmo fluxo; devolve quantas foram juntadas."""
    juntadas = 0
    for celula in nb.cells:
        if celula.cell_type != "code":
            continue
        novas = []
        for saida in celula.get("outputs", []):
            if (saida.output_type == "stream" and novas and novas[-1].output_type == "stream"
                    and novas[-1].name == saida.name):
                novas[-1].text += saida.text
                juntadas += 1
            else:
                novas.append(saida)
        celula.outputs = novas
    return juntadas


def main(caminhos: list[str]) -> None:
    for caminho in map(Path, caminhos):
        inicio = time.perf_counter()
        nb = nbformat.read(caminho, as_version=4)
        NotebookClient(nb, timeout=1200, kernel_name="python3",
                       resources={"metadata": {"path": str(caminho.parent)}}).execute()
        n = juntar_fluxos(nb)
        nbformat.write(nb, caminho)
        print(f"{caminho}: executado em {time.perf_counter() - inicio:.0f} s ({n} saídas de texto juntadas)")


if __name__ == "__main__":
    main(sys.argv[1:])
