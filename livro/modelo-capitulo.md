---
title: Modelo de capítulo (para autores)
label: modelo-capitulo
---

Esta página descreve a estrutura fixa de todos os capítulos. O arquivo [`modelos/capitulo-modelo.ipynb`](https://github.com/vnikoofard/calculo-numerico-livro/blob/main/modelos/capitulo-modelo.ipynb) é o ponto de partida para escrever um capítulo novo.

| # | Seção | Objetivo | Exemplo (cap. 3, Newton) |
| --- | --- | --- | --- |
| 1 | Problema de engenharia | Motivar com um caso real antes de qualquer teoria | Fator de atrito de Colebrook em tubulações |
| 2 | A ideia em uma imagem | Intuição geométrica, sem equações | A tangente "desliza" até o eixo, iteração por iteração |
| 3 | Formulação matemática | Deduzir o método e a análise de erro | Dedução por Taylor; convergência quadrática |
| 4 | À mão | Duas ou três iterações no papel | f(x) = x² − 2 com x₀ = 1 |
| 5 | Explore | Figuras com controles embutidos para variar parâmetros | Slider para x₀; bacias de convergência |
| 6 | Trabalhando com o agente | Especificar, gerar e verificar | Pedido modelo; ordem de convergência medida; comparação com SciPy |
| 7 | Quando falha | Casos em que o método ou o código erram | Raiz múltipla; ciclo em x³ − 2x + 2; derivada nula |

No fim de cada capítulo: uma tabela-resumo dos métodos, exercícios em três níveis (conceituais, de verificação com agente, projeto aberto) e leituras.

## Regras de estilo das figuras

- Use sempre `cnlivro.estilo`: `aplicar_estilo()` no início e `mostrar(fig)` para exibir cada figura.
- Cada cor tem um papel fixo (`CORES["exata"]`, `CORES["aproximacao"]`, `CORES["erro"]`...).
- Gráficos de erro ficam em escala log-log, com a reta da ordem teórica (`grafico_convergencia`).
- Eixos sempre com nome e unidade, em português e com vírgula decimal.
- Cada figura tem uma legenda de uma frase que diz o que o leitor deve notar.
- Interatividade com `com_controle(quadros, rotulo=...)`: sliders do Plotly que funcionam no site sem Python. No PDF aparece o quadro inicial. Não use `ipywidgets`.
- Quando ajudar, acrescente uma versão estática (pequenos múltiplos) para o PDF.
