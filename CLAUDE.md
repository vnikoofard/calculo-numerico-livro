# Instruções para agentes neste repositório

Livro *Cálculo Numérico na era dos agentes* (Jupyter Book 2 / MyST), em português do Brasil.

- Todo capítulo segue `livro/modelo-capitulo.md` e parte de `modelos/capitulo-modelo.ipynb` (7 seções fixas).
- Figuras: sempre Plotly via `cnlivro.estilo` (`aplicar_estilo()`, `mostrar(fig)`, cores de `CORES` por papel). Nunca Matplotlib nas páginas do livro. Vírgula decimal nos rótulos (`numero()`).
- Interatividade: use `com_controle` (sliders do Plotly, JavaScript puro). Funciona no site estático, sem Python, e no PDF vira o estado inicial. Não use `ipywidgets` nas páginas do livro: exige um kernel e, no site, mostra um aviso no lugar do controle. Quando ajudar, acrescente uma versão estática (pequenos múltiplos) para o PDF.
- Implementações de referência ficam em `src/cnlivro/` com testes em `tests/`; devolvem `Resultado` com histórico.
- Aprendizado de máquina e diferenciação automática: PyTorch.
- O texto é neutro quanto ao agente de IA (não citar um produto específico como obrigatório).
- Não copiar figuras do Chapra e Canale: redesenhar no estilo do livro.
- Antes de abrir PR: `pytest -q` e `jupyter book build --html --execute` sem erros.
