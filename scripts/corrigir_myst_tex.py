"""Corrige o código no PDF gerado pelo MyST (jupyter-book 2.1.7).

O exportador LaTeX do MyST passa o texto dos blocos de código pelo mesmo
"escape" usado em fórmulas: ``-`` vira `` -``, ``λ`` vira ``\\lambda``, ``°`` se
perde. Dentro de ``minted`` o texto é literal, então o certo é escrevê-lo sem
mudança. Este script troca, no pacote JavaScript instalado, a única chamada
que faz isso no tratamento dos blocos de código.

Uso (no CI, depois de instalar as dependências e antes de gerar o PDF):

    python scripts/corrigir_myst_tex.py

Falha com uma mensagem clara se o trecho não for encontrado (por exemplo,
numa versão nova do jupyter-book, que talvez já tenha corrigido o problema).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupyter_book

ARQUIVO = Path(jupyter_book.__file__).parent / "dist" / "jupyter-book.cjs"
ANTES = "    state.write(start);\n    state.text(node3.value, true);\n    state.write(end);"
DEPOIS = "    state.write(start);\n    state.write(node3.value);\n    state.write(end);"


def main() -> int:
    texto = ARQUIVO.read_text(encoding="utf-8")
    if DEPOIS in texto and ANTES not in texto:
        print("jupyter-book já corrigido.")
        return 0
    n = texto.count(ANTES)
    if n != 1:
        print(f"ERRO: trecho esperado encontrado {n} vezes em {ARQUIVO}. Verifique a versão do jupyter-book.")
        return 1
    ARQUIVO.write_text(texto.replace(ANTES, DEPOIS), encoding="utf-8")
    print(f"Corrigido: blocos de código do LaTeX agora são escritos sem escape ({ARQUIVO.name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
