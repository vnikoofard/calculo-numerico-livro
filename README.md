# Cálculo Numérico na era dos agentes

Livro de métodos numéricos para engenharia, em português, escrito para alunos que trabalham **com** agentes de IA: o agente escreve o código, e o aluno entende, especifica, verifica e critica.

- **Site (interativo):** https://vnikoofard.github.io/calculo-numerico-livro
- **PDF:** gerado automaticamente a cada atualização (aba *Actions* → *PDF do livro*)
- **Notas antigas:** [numerical_methods_using_python](https://github.com/vnikoofard/numerical_methods_using_python)

## Estrutura

| Pasta / arquivo | Conteúdo |
| --- | --- |
| `myst.yml` | Configuração do Jupyter Book e sumário do livro |
| `livro/` | Os capítulos (`.ipynb` ou `.md`) |
| `modelos/capitulo-modelo.ipynb` | Ponto de partida para escrever um capítulo |
| `src/cnlivro/` | Pacote do livro: estilo dos gráficos, verificação, implementações de referência |
| `tests/` | Testes do pacote |
| `.github/workflows/` | Publicação do site, geração do PDF e testes |

## Trabalhar no livro localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                               # testes do pacote
jupyter book start --execute            # site local em http://localhost:3000
jupyter book build --pdf --execute      # PDF (precisa de LaTeX instalado)
```

Para que as figuras apareçam no PDF, defina `CNLIVRO_PNG=1` antes de construir.

## Publicação

O site é publicado no GitHub Pages pelo workflow `publicar.yml`. Na primeira vez, é preciso ativar em **Settings → Pages → Source: GitHub Actions**.
