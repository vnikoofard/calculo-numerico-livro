#!/usr/bin/env bash
# Gera o PDF do livro em exports/calculo-numerico.pdf (usado pelo CI).
#
# 1. corrige o exportador LaTeX do MyST (código sem escape; ver corrigir_myst_tex.py);
# 2. executa os capítulos e grava as saídas NOS PRÓPRIOS NOTEBOOKS, juntando as
#    saídas de texto picotadas (ver executar_capitulos.py);
# 3. gera o PDF com o modelo latex/livro, sem executar de novo.
#
# Atenção: o passo 2 altera livro/*.ipynb. Localmente, rode numa cópia ou
# desfaça depois com `git checkout livro/`.
set -euo pipefail
export CNLIVRO_PNG=1
export BROWSER_PATH="${BROWSER_PATH:-$(which google-chrome || which chromium || true)}"
python scripts/corrigir_myst_tex.py
python scripts/executar_capitulos.py livro/*.ipynb
jupyter book build --pdf
