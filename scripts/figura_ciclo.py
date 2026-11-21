"""Gera a figura estática do ciclo de trabalho com o agente (cap. 0): livro/figuras/ciclo-trabalho.png.

Uso: python scripts/figura_ciclo.py
"""
import plotly.graph_objects as go

from cnlivro.estilo import CORES, aplicar_estilo

aplicar_estilo()
etapas = [
    ("1. Entender", "a ideia do método:<br>desenho, hipóteses,<br>ordem de convergência"),
    ("2. Especificar", "o pedido ao agente:<br>entradas, saídas, parada,<br>falhas, testes"),
    ("3. Verificar", "o código gerado:<br>resíduo, solução conhecida,<br>ordem medida, biblioteca"),
    ("4. Criticar", "os limites:<br>onde o método falha<br>e o que usar no lugar"),
]
cores = [CORES["aproximacao"], CORES["secundaria"], CORES["destaque"], CORES["erro"]]
fig = go.Figure()
for i, ((titulo, texto), cor) in enumerate(zip(etapas, cores)):
    x0 = i * 2.6
    # cartão com faixa colorida no topo
    fig.add_shape(type="rect", x0=x0, x1=x0 + 2.1, y0=0, y1=1.6, line=dict(color=cor, width=2), fillcolor="white",
                  layer="below")
    fig.add_shape(type="rect", x0=x0, x1=x0 + 2.1, y0=1.12, y1=1.6, line=dict(color=cor, width=2), fillcolor=cor)
    fig.add_annotation(x=x0 + 1.05, y=1.36, text=f"<b>{titulo}</b>", showarrow=False, font=dict(size=17, color="white"))
    fig.add_annotation(x=x0 + 1.05, y=0.56, text=texto, showarrow=False, font=dict(size=13, color="#333333"))
    if i < 3:
        fig.add_annotation(x=x0 + 2.55, y=0.8, ax=x0 + 2.15, ay=0.8, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor=CORES["referencia"], text="")
# seta de retorno, por baixo dos cartões
fig.add_shape(type="path", path="M 8.85 -0.02 L 8.85 -0.3 L 1.05 -0.3 L 1.05 -0.06",
              line=dict(color=CORES["referencia"], width=2, dash="dot"))
fig.add_annotation(x=1.05, y=-0.02, ax=1.05, ay=-0.12, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                   arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=CORES["referencia"], text="")
fig.add_annotation(x=4.95, y=-0.47, showarrow=False, font=dict(size=13, color=CORES["referencia"]),
                   text="a crítica volta ao entendimento: um teste que falha ensina algo sobre o método")
fig.update_xaxes(visible=False, range=[-0.15, 10.1])
fig.update_yaxes(visible=False, range=[-0.65, 1.7])
fig.update_layout(width=980, height=290, margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
                  paper_bgcolor="white", plot_bgcolor="white")
fig.write_image("livro/figuras/ciclo-trabalho.png", scale=2)
print("livro/figuras/ciclo-trabalho.png")
