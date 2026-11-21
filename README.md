<p align="center">
  <img src=".github/assets/capa-livia-uerj-fat.png" alt="LIVIA, Universidade do Estado do Rio de Janeiro (UERJ) e Faculdade de Tecnologia (FAT), UERJ-Resende" width="100%">
</p>

<p align="center">
  <a href="https://vnikoofard.github.io/calculo-numerico-livro/"><b>📖 Ler o livro no site</b></a>
  &nbsp;·&nbsp;
  <a href="https://vnikoofard.github.io/calculo-numerico-livro/calculo-numerico.pdf"><b>📄 Baixar o PDF</b></a>
</p>

# Cálculo Numérico na era dos agentes

*Métodos numéricos para engenharia, com Python e agentes de IA*

**Vahid Nikoofard** · Faculdade de Tecnologia (FAT), Universidade do Estado do Rio de Janeiro (UERJ), Resende

## Sobre este livro

Este livro nasceu da minha experiência como professor das disciplinas **Cálculo Numérico 1** e **Cálculo Numérico 2** na Faculdade de Tecnologia (FAT) da UERJ. Ele reescreve as notas de aula que preparei para essas turmas, a partir do que funcionou, e do que não funcionou, em sala.

Escrevi esta versão porque a forma de estudar cálculo numérico mudou. Hoje, meus alunos têm à mão agentes de IA que escrevem em segundos o código de qualquer método do curso: bisseção, eliminação de Gauss, Runge-Kutta. O que o agente não faz é responder pelo resultado. Um código numérico pode rodar sem erro, devolver números plausíveis e estar errado, e quem assina um projeto de engenharia precisa saber **por que** o número está certo.

Por isso, o foco do livro não é programar os métodos, e sim **entendê-los, especificá-los, verificar e criticar** o código que o agente devolve. Cada capítulo parte de um problema de engenharia, constrói a intuição com figuras interativas, deduz o método, faz algumas iterações à mão, mostra um código típico de um agente e o verifica, e termina com os casos em que o método, ou o código, falha. O livro não depende de um agente específico: vale para qualquer um.

Todo capítulo segue as mesmas sete seções:

1. **Problema de engenharia**: um caso real que motiva o método.
2. **A ideia em uma imagem**: a intuição geométrica, antes de qualquer equação.
3. **Formulação matemática**: a dedução e a análise de erro.
4. **À mão**: duas ou três iterações no papel.
5. **Explore**: gráficos e controles interativos para variar os parâmetros.
6. **Trabalhando com o agente**: especificar, gerar e verificar o código.
7. **Quando falha**: os casos em que o método, ou o código do agente, erra.

Os capítulos são notebooks Jupyter, que podem ser lidos no navegador (com o código executável ali mesmo), abertos no Google Colab ou baixados em PDF. As implementações de referência dos métodos ficam no pacote `cnlivro`, que acompanha o livro, com testes automáticos. As redes neurais e a diferenciação automática usam PyTorch.

A leitura de apoio continua sendo o livro *Numerical Methods for Engineers*, de Steven Chapra e Raymond Canale. As minhas notas antigas estão no repositório [numerical_methods_using_python](https://github.com/vnikoofard/numerical_methods_using_python).

## Sumário

**Como usar este livro com um agente de IA**: o ciclo de trabalho seguido em todos os capítulos.

**Parte I: Fundamentos**

1. Modelagem e o ciclo da computação científica
2. Erros e aritmética de ponto flutuante

**Parte II: Equações e sistemas**

3. Raízes de equações
4. Sistemas de equações lineares
5. Sistemas não lineares e otimização

**Parte III: Dados e aproximação**

6. Interpolação
7. Mínimos quadrados e ajuste de modelos
8. Autovalores, autovetores e SVD
9. Do ajuste de curvas ao aprendizado de máquina

**Parte IV: Cálculo**

10. Derivação numérica e diferenciação automática
11. Integração numérica

**Parte V: Equações diferenciais**

12. EDOs: problemas de valor inicial
13. Estabilidade e rigidez
14. Problemas de contorno
15. EDPs com diferenças finitas
16. Método das linhas e bibliotecas

**Apêndices**

- A. Incerteza e sensibilidade
- B. Desempenho e GPU
- C. Referência rápida do pacote `cnlivro`

## Estrutura do repositório

| Pasta / arquivo | Conteúdo |
| --- | --- |
| `livro/` | Os capítulos (notebooks `.ipynb` e páginas `.md`) e as figuras |
| `myst.yml` | Configuração do Jupyter Book e sumário do livro |
| `src/cnlivro/` | Pacote do livro: estilo dos gráficos, verificação e implementações de referência |
| `tests/` | Testes do pacote |
| `modelos/capitulo-modelo.ipynb` | Ponto de partida para escrever um capítulo |
| `latex/livro/` | Modelo LaTeX do PDF |
| `scripts/` | Geração do PDF e de figuras |
| `.github/workflows/` | Testes, site e PDF automáticos |

## Instalação local

Para executar os capítulos e rodar os testes no seu computador:

```bash
git clone https://github.com/vnikoofard/calculo-numerico-livro
cd calculo-numerico-livro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                               # testes do pacote
jupyter book start --execute            # site local em http://localhost:3000
```

Para gerar o PDF (é preciso ter LaTeX instalado), use `bash scripts/construir_pdf.sh`. O script executa os capítulos e grava as saídas nos próprios notebooks; por isso, rode-o numa cópia do repositório ou desfaça as mudanças depois com `git checkout livro/`.

Para usar só o pacote `cnlivro`, sem o livro:

```bash
pip install git+https://github.com/vnikoofard/calculo-numerico-livro
```
