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

Os demais módulos (sistemas lineares, interpolação, EDOs...) serão portados do
``cn_utils.py`` das notas antigas à medida que os capítulos forem reescritos.
"""

__version__ = "0.1.0"
