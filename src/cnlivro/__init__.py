"""cnlivro: pacote de apoio do livro *Cálculo Numérico na era dos agentes*.

Módulos
-------
estilo       tema gráfico único (Plotly), paleta por papel e gráficos de convergência
verificacao  ferramentas para verificar código numérico (ordem observada, comparações)
raizes       implementações de referência dos métodos de raízes (cap. 3)
lineares     sistemas lineares: LU com pivotamento, Jacobi, Gauss-Seidel, gradiente conjugado (cap. 4)
otimizacao   Newton para sistemas, descida pelo gradiente, BFGS (cap. 5)
interpolacao diferenças divididas, Lagrange baricêntrica, nós de Chebyshev, spline natural (cap. 6)
ajuste       equações normais, QR, R², Gauss-Newton (cap. 7)
autovalores  método da potência (direto e inverso), prédio de cisalhamento (cap. 8)
redes        rede totalmente conectada, padronizador e laço de treino em PyTorch (cap. 9)
derivacao    diferenças finitas, Richardson, passo complexo, derivada de dados (cap. 10)
integracao   trapézio, Simpson, Gauss-Legendre, Romberg, Simpson adaptativo, Monte Carlo (cap. 11)
edo          Euler, RK4, RK45 adaptativo, implícitos, Verlet, tiro, diferenças finitas, Schrödinger (caps. 12-14)
edp          advecção-difusão (upwind, Lax-Wendroff), calor (método θ), Poisson 2D, Burgers, laplacianos, elementos finitos 1D (caps. 15-16)
incerteza    Monte Carlo com erro padrão, intervalo de Wilson, método delta, gradiente com PyTorch (apêndice A)
desempenho   medição de tempo, dispositivo do PyTorch, algoritmo de Thomas (um sistema e em lote) (apêndice B)

O apêndice C traz uma referência rápida de todas as funções, com os equivalentes no NumPy, no SciPy e no PyTorch.
"""

__version__ = "0.1.0"
