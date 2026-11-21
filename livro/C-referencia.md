---
title: "Referência rápida do pacote cnlivro"
---

Os capítulos usam um pacote pequeno, o `cnlivro`, com as **implementações de referência** dos métodos do livro e as ferramentas de figura e de verificação. Este apêndice reúne, num só lugar, o que cada função faz, em que capítulo ela aparece e qual é a função equivalente nas bibliotecas que se usam em trabalho de verdade (NumPy, SciPy, PyTorch).

## Para que serve o pacote

As funções do `cnlivro` foram escritas para serem **lidas**: são curtas, seguem a notação dos capítulos e guardam o histórico das iterações, para que se possa ver o método funcionando. Elas têm testes automáticos (pasta `tests/` do repositório) e servem para três coisas:

1. **Entender** um método: ler o código ao lado da formulação do capítulo.
2. **Verificar** o código de um agente: comparar a resposta dele com a da função de referência, nos mesmos dados (a seção 6 de cada capítulo faz isso).
3. **Explorar**: produzir rapidamente as curvas de convergência e os gráficos dos capítulos.

Elas **não** foram escritas para desempenho nem para cobrir todos os casos. Para resolver um problema de engenharia, use a biblioteca de produção indicada na última coluna das tabelas, e use o `cnlivro` para conferir a resposta num caso pequeno.

## Instalação

Cada capítulo começa com uma célula que instala o pacote quando ele falta, no Colab, num computador ou no navegador (JupyterLite). Para instalar à mão:

```bash
pip install git+https://github.com/vnikoofard/calculo-numerico-livro
```

Para trabalhar no repositório (alterar o pacote e rodar os testes):

```bash
git clone https://github.com/vnikoofard/calculo-numerico-livro
cd calculo-numerico-livro
pip install -e ".[livro,dev]"
pytest -q
```

## Convenções

**Resultados com histórico.** Os métodos iterativos devolvem um objeto de resultado, e não só o número final. Todos têm o valor calculado, uma indicação de sucesso e o histórico:

| Objeto | Módulo | Campos principais |
| --- | --- | --- |
| `Resultado` | `raizes` | `raiz`, `convergiu`, `iteracoes`, `historico` (os $x_k$), `erros` (erro relativo aproximado) |
| `ResultadoVetorial` | `lineares`, `otimizacao` | `x`, `convergiu`, `iteracoes`, `historico`, `erros` |
| `TabelaExtrapolacao` | `derivacao` | `valor`, `tabela` (a tabela de Richardson), `passos` |
| `ResultadoIntegral` | `integracao` | `valor`, `convergiu`, `avaliacoes`, `historico`, `erros` |
| `ResultadoTreino` | `redes` | `modelo`, `epocas`, `historico`, `historico_validacao` |
| `ResultadoEDO` | `edo` | `t`, `y`, `avaliacoes`, `sucesso`, `motivo`, `rejeitados`, `historico` |
| `ResultadoEDP` | `edp` | `x`, `t`, `u`, `passos`, `sucesso`, `motivo`, `historico` |
| `ResultadoMC` | `incerteza` | `media`, `desvio`, `erro_padrao`, `amostras`, `n`, `historico` |
| `ResultadoTempo` | `desempenho` | `mediana`, `minimo`, `tempos`, `valor` |

Um método que não converge **não** lança erro: ele devolve `convergiu=False` (ou `sucesso=False`, com o `motivo`). Confira sempre esse campo antes de usar o valor. É exatamente o cuidado que falta em muitos códigos gerados às pressas.

**Nomes em português, notação dos capítulos.** Os argumentos seguem as fórmulas do texto: `f` e `df` para a função e a derivada, `x0` para o chute inicial, `tol` para a tolerância, `h` para o passo, `t_span` para o intervalo de integração.

**Funções vetorizadas.** Onde faz sentido (quadraturas, EDOs, Monte Carlo), as funções do usuário recebem vetores do NumPy e devem devolver vetores.

## Figuras: `cnlivro.estilo`

| Função | O que faz |
| --- | --- |
| `aplicar_estilo()` | registra o tema do livro no Plotly (fontes, grade, cores) |
| `mostrar(fig)` | mostra a figura interativa e, ao gerar o PDF, também uma versão estática |
| `com_controle(quadros, rotulo, ativo, titulos)` | junta várias figuras numa só, com um controle deslizante; funciona no site sem Python |
| `numero(valor, casas)` | formata números com vírgula decimal, para títulos e rótulos |
| `grafico_convergencia(h, erro, ordens)` | gráfico log-log do erro contra o passo, com retas de ordem teórica |
| `grafico_iteracoes(resultados)` | erro por iteração de vários métodos, em escala logarítmica |
| `reta_de_ordem(x_min, x_max, y_ref, ordem)` | uma reta de inclinação dada, para gráficos log-log |
| `CORES` | paleta por papel: `exata`, `aproximacao`, `erro`, `referencia`, `destaque`, `secundaria`, `terciaria` |

## Verificação: `cnlivro.verificacao`

| Função | O que faz | Equivalente |
| --- | --- | --- |
| `ordem_observada(h, erro)` | ordem $p$ de convergência a partir de erro $\approx C h^p$ (ajuste em escala log) | `np.polyfit(np.log(h), np.log(erro), 1)` |
| `ordem_iterativa(historico, raiz)` | ordem $q$ de um método iterativo, de $\lvert e_{k+1}\rvert \approx C \lvert e_k\rvert^q$ | — |
| `compara(valor, referencia, rtol, atol)` | diz se dois resultados concordam e explica a diferença | `np.testing.assert_allclose`, `pytest.approx` |

## Métodos, por capítulo

### Raízes: `cnlivro.raizes` (capítulo [3](./03-raizes.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `bissecao(f, a, b)` | bisseção num intervalo com troca de sinal | `scipy.optimize.bisect` |
| `falsa_posicao(f, a, b)` | falsa posição (*regula falsi*) | `scipy.optimize.brentq` (intervalo, mais robusto) |
| `ponto_fixo(g, x0)` | iteração $x_{k+1} = g(x_k)$ | `scipy.optimize.fixed_point` |
| `newton(f, df, x0, m=1)` | Newton-Raphson; `m` é a multiplicidade da raiz | `scipy.optimize.newton(f, x0, fprime=df)` |
| `secante(f, x0, x1)` | secante | `scipy.optimize.newton(f, x0)` (sem derivada) |

### Sistemas lineares: `cnlivro.lineares` (capítulo [4](./04-sistemas-lineares.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `substituicao_regressiva(U, b)`, `substituicao_progressiva(L, b)` | sistemas triangulares | `solve_triangular` (`scipy.linalg`) |
| `lu(A)` | LU com pivotamento parcial, $PA = LU$ | `scipy.linalg.lu_factor` (ou `scipy.linalg.lu`, que devolve $A = PLU$: o $P$ é o transposto) |
| `resolve_lu(P, L, U, b)` | resolve a partir da fatoração | `scipy.linalg.lu_solve` |
| `eliminacao_gauss(A, b, pivotamento=True)` | eliminação de Gauss | `np.linalg.solve` |
| `jacobi(A, b)`, `gauss_seidel(A, b)` | métodos iterativos clássicos | `scipy.sparse.linalg` (`cg`, `gmres`, `bicgstab`), com pré-condicionador |
| `gradiente_conjugado(A, b)` | gradiente conjugado, $A$ simétrica positiva definida | `scipy.sparse.linalg.cg` |
| `placa_laplace(N, ...)` | monta o sistema esparso da placa do capítulo | `scipy.sparse.diags`, `scipy.sparse.kron` |

### Otimização: `cnlivro.otimizacao` (capítulo [5](./05-otimizacao.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `newton_sistema(F, J, x0)` | Newton para $\mathbf F(\mathbf x) = \mathbf 0$ | `scipy.optimize.root` |
| `busca_armijo(f, x, fx, g, d)` | busca linear com retrocesso | `scipy.optimize.line_search` (condições de Wolfe) |
| `gradiente_descendente(f, grad, x0)` | descida pelo gradiente | `torch.optim.SGD` |
| `bfgs(f, grad, x0)` | quase-Newton BFGS | `minimize(method="BFGS")` (`scipy.optimize`) |
| `rosenbrock(x)`, `grad_rosenbrock(x)` | função de teste de Rosenbrock | `scipy.optimize.rosen`, `rosen_der` |

### Interpolação: `cnlivro.interpolacao` (capítulo [6](./06-interpolacao.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `diferencas_divididas(x, y)`, `newton_avalia(coef, x_nos, t)` | forma de Newton do polinômio interpolador | `KroghInterpolator` (`scipy.interpolate`) |
| `lagrange(x_nos, y_nos, t)` | fórmula baricêntrica | `BarycentricInterpolator` (`scipy.interpolate`) |
| `nos_chebyshev(n, a, b)` | nós de Chebyshev em $[a, b]$ | `chebpts1` (`numpy.polynomial.chebyshev`), em $[-1, 1]$ |
| `spline_cubica_natural(x, y)` | spline cúbica natural | `scipy.interpolate.CubicSpline(x, y, bc_type="natural")` |

### Mínimos quadrados: `cnlivro.ajuste` (capítulo [7](./07-minimos-quadrados.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `equacoes_normais(A, b)` | resolve $A^\mathsf T A\mathbf x = A^\mathsf T\mathbf b$ (eleva $\kappa$ ao quadrado: só para comparação) | — |
| `minimos_quadrados_qr(A, b)` | mínimos quadrados por QR | `np.linalg.lstsq`, `scipy.linalg.lstsq` |
| `r2(y, y_ajustado)` | coeficiente de determinação | `sklearn.metrics.r2_score` |
| `gauss_newton(residuo, jacobiana, p0)` | ajuste não linear | `scipy.optimize.least_squares`, `scipy.optimize.curve_fit` |

### Autovalores: `cnlivro.autovalores` (capítulo [8](./08-autovalores-svd.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `metodo_potencia(A, deslocamento, inversa)` | potência, e potência inversa com deslocamento | `scipy.sparse.linalg.eigsh` / `eigs` (com `sigma` para o deslocamento) |
| `predio_cisalhamento(massas, rigidezes)` | matrizes $M$ e $K$ do prédio do capítulo | problema generalizado: `scipy.linalg.eigh(K, M)` |

### Aprendizado de máquina: `cnlivro.redes` (capítulo [9](./09-aprendizado-de-maquina.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `mlp(entradas, saidas, largura, camadas)` | rede totalmente conectada em PyTorch | `torch.nn.Sequential` |
| `Padronizador(dados)` | média 0 e desvio 1 por coluna, e a volta | `StandardScaler` (`sklearn.preprocessing`) |
| `treinar(modelo, x, y, ...)` | treino por erro quadrático médio com Adam, com validação | laço com `torch.optim.Adam`; bibliotecas como PyTorch Lightning |

### Derivação: `cnlivro.derivacao` (capítulo [10](./10-derivacao.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `progressiva`, `regressiva`, `centrada`, `segunda_centrada` | fórmulas de diferenças finitas | `derivative` (`scipy.differentiate`), com passo escolhido automaticamente |
| `passo_complexo(f, x)` | derivada pelo passo complexo | — |
| `richardson(f, x, h, niveis)` | extrapolação de Richardson | `derivative` (`scipy.differentiate`) |
| `derivada_dados(x, y)` | derivada de dados com espaçamento qualquer | `np.gradient(y, x, edge_order=2)` |
| — | derivada exata de código | diferenciação automática: `torch.autograd`, `torch.func.grad` |

### Integração: `cnlivro.integracao` (capítulo [11](./11-integracao.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `trapezio(f, a, b, n)`, `simpson(f, a, b, n)` | regras compostas com $n$ subintervalos | `scipy.integrate.trapezoid`, `scipy.integrate.simpson` (sobre amostras) |
| `trapezio_dados(x, y)` | trapézio para dados | `np.trapezoid` |
| `gauss_legendre(f, a, b, n)` | Gauss-Legendre com $n$ pontos | `scipy.integrate.fixed_quad`; nós em `leggauss` (`np.polynomial.legendre`) |
| `romberg(f, a, b, tol)`, `simpson_adaptativo(f, a, b, tol)` | integração com controle do erro | `scipy.integrate.quad` |
| `monte_carlo(f, limites, n)` | Monte Carlo numa caixa, com erro padrão | `scipy.integrate.qmc_quad` (quase Monte Carlo) |

### Equações diferenciais ordinárias: `cnlivro.edo` (capítulos [12](./12-edo-valor-inicial.ipynb) a [14](./14-problemas-de-contorno.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `passo_fixo(f, t_span, y0, h, metodo)` | Euler, Heun ou RK4 com passo fixo (`metodo="euler"`, `"heun"`, `"rk4"`) | `scipy.integrate.solve_ivp` (passo adaptativo) |
| `rk45(f, t_span, y0, rtol, atol)` | Runge-Kutta adaptativo de Dormand-Prince | `solve_ivp(method="RK45")` |
| `implicito(f, t_span, y0, h, metodo)` | Euler implícito e trapézio implícito | `solve_ivp(method="Radau")` ou `"BDF"` |
| `adams_bashforth2(f, t_span, y0, h)` | método de passo múltiplo | `solve_ivp(method="LSODA")` |
| `verlet(aceleracao, t_span, q0, v0, h)` | Verlet de velocidade (simplético) | não há no SciPy; integradores simpléticos estão em bibliotecas de dinâmica |
| `tiro(f, x_span, ya, residuo, s0, s1)` | método do tiro | `scipy.optimize.root_scalar` com `solve_ivp` |
| `contorno_dif_finitas(f, x_span, ya, direita, n)` | diferenças finitas com Newton | `scipy.integrate.solve_bvp` |
| `autoestados(V, x_span, n, k)` | níveis de energia da equação de Schrödinger 1D | `eigh_tridiagonal` (`scipy.linalg`) |

### Equações diferenciais parciais: `cnlivro.edp` (capítulos [15](./15-edp-diferencas-finitas.ipynb) e [16](./16-metodo-das-linhas.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `passo_maximo(dx, u, D, esquema)` | maior passo de tempo estável (von Neumann) | — |
| `adveccao_difusao(c0, x, u, D, t_final, esquema)` | advecção-difusão explícita (upwind, Lax-Wendroff, centrado) | método das linhas com `solve_ivp`; bibliotecas de volumes finitos (Clawpack, FiPy) |
| `calor_theta(u0, x, alpha, t_final, dt, theta)` | calor 1D pelo método $\theta$ (Euler explícito, Crank-Nicolson, implícito) | método das linhas com `solve_ivp(method="BDF")` |
| `laplaciano_1d(n, dx)`, `laplaciano_2d(nx, ny, dx, dy)` | matrizes esparsas das segundas diferenças | `scipy.sparse.diags`, `scipy.sparse.kron` |
| `poisson_2d(f, g, Lx, Ly, nx, ny, metodo)` | Poisson 2D: direto, Jacobi, Gauss-Seidel, SOR | `scipy.sparse.linalg.spsolve`; multigrid (PyAMG) |
| `burgers(u0, x, t_final, dt, forma)` | Burgers sem viscosidade, forma conservativa ou não | bibliotecas de volumes finitos (Clawpack) |
| `fem_1d(k, f, nos, ua, direita)` | elementos finitos lineares em 1D | scikit-fem, FEniCSx |

### Incerteza: `cnlivro.incerteza` (apêndice [A](./A-incerteza.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `monte_carlo(modelo, entradas)` | avalia o modelo em todas as amostras; média, desvio e erro padrão | NumPy e `scipy.stats` (amostragem: `scipy.stats.qmc`) |
| `probabilidade(eventos, confianca)` | probabilidade de um evento com o intervalo de Wilson | `scipy.stats.binomtest(k, n).proportion_ci(method="wilson")` |
| `propagacao_linear(f, x0, sigmas)` | método delta, com derivadas por diferenças | pacote `uncertainties` |
| `gradiente_torch(f, x0)` | gradiente por diferenciação automática | `torch.func.grad`, `jacobian` (`torch.autograd.functional`) |
| — | análise de sensibilidade global (índices de Sobol) | SALib |

### Desempenho: `cnlivro.desempenho` (apêndice [B](./B-desempenho-gpu.ipynb))

| Função | O que faz | Equivalente de produção |
| --- | --- | --- |
| `cronometrar(f, *args, repeticoes, aquecimento, sincronizar)` | mediana de várias execuções, sem a primeira, com sincronização da GPU | `timeit` e `%timeit`; `torch.utils.benchmark` |
| `dispositivo()` | `"cuda"`, `"mps"` ou `"cpu"` | — |
| `thomas(a, b, c, d)` | sistema tridiagonal pelo algoritmo de Thomas | `scipy.linalg.solve_banded` |
| `thomas_lote(a, b, c, d)` | lote de sistemas tridiagonais, vetorizado sobre o lote | um laço com `solve_banded`; na GPU, solvers tridiagonais em lote (cuSPARSE) |

## Usando o pacote para verificar um agente

O padrão que os capítulos seguem cabe em poucas linhas. Peça ao agente uma solução; resolva o mesmo problema, **pequeno**, com a função de referência; e compare:

```python
from cnlivro.edo import rk45
from cnlivro.verificacao import compara

r_ref = rk45(f, (0, 10), y0, rtol=1e-10, atol=1e-12)   # referência, com tolerância apertada
y_agente = solucao_do_agente(f, (0, 10), y0)            # o código devolvido pelo agente
compara(y_agente[-1], r_ref.y[-1], rtol=1e-6)
```

Se os dois discordam, um deles está errado, e o histórico do resultado de referência (passos aceitos e rejeitados, correções de Newton, erros por iteração) costuma mostrar onde olhar. Se concordam, aumente o problema e repita a comparação num caso em que a resposta exata é conhecida: a concordância entre dois códigos não prova que ambos estão certos, só que não erraram da mesma forma.
