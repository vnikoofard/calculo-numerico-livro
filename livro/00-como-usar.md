# Como usar este livro com um agente de IA

Hoje, um agente de IA escreve em segundos o código de qualquer método deste livro: bisseção, eliminação de Gauss, Runge-Kutta. Então para que estudar cálculo numérico?

Porque **o agente não responde pelo resultado. Você responde.** Um código numérico pode rodar sem erro, devolver números plausíveis e estar errado. Ele pode usar um método que não serve para o problema, perder precisão sem avisar ou convergir para a resposta certa pelo motivo errado. Quem assina o projeto de uma ponte, de uma tubulação ou de um circuito precisa saber **por que** o número está certo. Este livro ensina isso.

## O ciclo de trabalho

Em todos os capítulos, o trabalho com o agente segue o mesmo ciclo de quatro passos.

```{figure} figuras/ciclo-trabalho.png
:name: fig-ciclo-trabalho
:width: 100%
:alt: Os quatro passos do ciclo de trabalho com o agente: entender, especificar, verificar e criticar, com a crítica voltando ao entendimento.

O ciclo de trabalho com o agente.
```

1. **Entender.** Antes de pedir qualquer código, você precisa saber explicar o método com um desenho e uma frase. Também precisa saber em que hipóteses ele funciona e quão rápido deve convergir. Sem isso não há como escrever um bom pedido nem como julgar a resposta.
2. **Especificar.** Um pedido vago produz código vago. O pedido diz o que entra, o que sai, como parar, o que fazer em caso de falha e como testar.
3. **Verificar.** Você confirma, com testes e gráficos, que o código faz o que a teoria diz. A seção seguinte traz a lista de verificações usada no livro inteiro.
4. **Criticar.** Você procura os casos em que o método, ou o código, falha e decide o que usar no lugar.

## Anatomia de um bom pedido

Um pedido bom para código numérico tem seis partes. Use este modelo como ponto de partida:

```text
TAREFA:     Implemente em Python a função <nome>(<argumentos>) que <o que faz>.
MÉTODO:     Use <método>, porque <motivo>. Não use <alternativa a evitar>.
ENTRADAS:   <tipos, unidades, faixas válidas>.
SAÍDAS:     <o que devolve>, incluindo o histórico das iterações.
PARADA:     <critério, com tolerância relativa e número máximo de iterações>.
FALHAS:     Lance ValueError se <condição inválida>. Nunca devolva um resultado
            silenciosamente quando não convergir.
TESTES:     Escreva testes com pytest que verifiquem <solução conhecida>,
            <comparação com biblioteca> e <ordem de convergência esperada>.
```

Dois detalhes fazem muita diferença:

- **Peça o histórico das iterações**, e não só o resultado final. É com ele que se mede a ordem de convergência, o teste mais poderoso do livro.
- **Peça os testes junto com o código.** Depois, leia os testes. Um teste que só confere se a função "roda" não verifica nada.

## Lista de verificação

Todo código numérico, escrito por você ou por um agente, passa por estes testes antes de ser usado:

| # | Verificação | O que detecta | Exemplo no livro |
| --- | --- | --- | --- |
| 1 | **Resíduo**: substituir o resultado na equação original | resultado que não resolve o problema | bisseção convergindo para um polo (cap. 3) |
| 2 | **Solução conhecida**: um caso com resposta analítica | erros grosseiros de fórmula | paraquedista com arrasto linear (cap. 1) |
| 3 | **Ordem de convergência medida**: gráfico log-log do erro | método implementado de forma errada, mesmo com resultado certo | Newton com a derivada errada (cap. 3) |
| 4 | **Comparação com biblioteca**: SciPy, NumPy | erros sutis e casos difíceis | fator de atrito contra o `brentq` (cap. 3) |
| 5 | **Casos limite**: zero, valores enormes, entradas inválidas | divisões por zero, overflow, perda de precisão | variância com média grande (cap. 2) |
| 6 | **Invariantes físicas**: conservação de massa, energia, simetria | modelos ou discretizações inconsistentes | balanço de energia (caps. 1 e 13) |

## Erros típicos de código gerado por IA

Estes erros aparecem com frequência em código numérico gerado automaticamente. Todos produzem código que roda sem erro:

- **Derivada ou fórmula ligeiramente errada.** Um fator esquecido, como o $\ln 10$ na derivada de $\log_{10}$, ou um sinal trocado. O método muitas vezes ainda converge, só que mais devagar.
- **Critério de parada inadequado.** Usar tolerância absoluta onde deveria ser relativa, ou parar quando $|f(x)|$ é pequeno sem olhar $x$.
- **Comparação de floats com `==`.** Um evento que deveria acontecer em `t == 10.0` nunca acontece, porque `t` vale `9.99999999`.
- **Fórmula matematicamente correta, numericamente instável.** Por exemplo, a variância por $E[x^2] - E[x]^2$ ou a fórmula de Bhaskara com cancelamento.
- **Unidades misturadas.** Milímetros com metros, graus com radianos.
- **Método inadequado ao problema.** Euler explícito num problema rígido, inversão de matriz para resolver um sistema linear.

## O ambiente

Você pode ler e executar este livro de três formas:

- **No site.** As figuras são interativas: dá zoom, mostra os valores ao passar o mouse e tem sliders para variar parâmetros, sem instalar nada. O botão de executar no topo da página roda o código no seu navegador.
- **No Google Colab.** Cada capítulo tem um link no topo. Você pode editar e rodar tudo, e pedir ajuda ao assistente do Colab ou a outro agente.
- **No seu computador.** Clone o repositório e instale as dependências com `pip install -r requirements.txt`. Este é o melhor ambiente para trabalhar com agentes de código.

O pacote `cnlivro`, instalado junto com o livro, traz o estilo dos gráficos, as ferramentas de verificação (`ordem_observada`, `ordem_iterativa`, `compara`) e implementações de referência dos métodos.

## Sobre honestidade acadêmica

Usar um agente neste curso não é trapaça: é o esperado. Mas vale a mesma regra de qualquer trabalho de engenharia: **você assina o que entrega**. Nas atividades, entregue o pedido que fez ao agente, o código, os testes e a sua análise dos resultados. A nota vem da qualidade da especificação e da verificação, não da quantidade de código.
