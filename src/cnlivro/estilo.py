"""Estilo gráfico único do livro.

Todas as figuras do livro passam por este módulo, para que tenham a mesma
aparência na versão web (interativa) e na versão em PDF (estática).

Uso típico no início de cada capítulo::

    from cnlivro.estilo import aplicar_estilo, mostrar, CORES
    aplicar_estilo()

    fig = go.Figure(...)
    mostrar(fig)

A função :func:`mostrar` envia ao notebook a figura interativa do Plotly e,
quando a variável de ambiente ``CNLIVRO_PNG=1`` está definida (como na
construção do PDF), também uma imagem PNG da mesma figura. O Jupyter Book usa
a versão interativa no site e a imagem no PDF.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paleta: cada cor tem um PAPEL fixo em todo o livro.
# Baseada na paleta Okabe-Ito, segura para daltonismo.
# ---------------------------------------------------------------------------
CORES = {
    "exata": "#222222",        # solução exata ou de referência
    "aproximacao": "#0072B2",  # resultado do método numérico
    "erro": "#D55E00",         # erros e resíduos
    "referencia": "#999999",   # retas de ordem teórica, guias
    "destaque": "#009E73",     # ponto atual, raiz, valor de interesse
    "secundaria": "#CC79A7",   # segundo método numa comparação
    "terciaria": "#E69F00",    # terceiro método numa comparação
}

# Ordem das cores quando há várias séries sem papel definido.
SEQUENCIA = [
    CORES["aproximacao"],
    CORES["secundaria"],
    CORES["terciaria"],
    CORES["destaque"],
    CORES["erro"],
    "#56B4E9",
]

FONTE = "Inter, 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif"
LARGURA_PADRAO = 720
ALTURA_PADRAO = 420

# Configuração das figuras no site: ``responsive`` faz o Plotly redesenhar a
# figura quando a largura da página muda (janela redimensionada, celular).
CONFIG_WEB = {"responsive": True, "displaylogo": False}

# Figuras mais largas que LARGURA_PADRAO não encolhem abaixo desta fração da
# largura original (ver mostrar()).
FRACAO_MINIMA = 0.6


def _template() -> go.layout.Template:
    eixo = dict(
        showline=True,
        linecolor="#444444",
        linewidth=1,
        ticks="outside",
        tickcolor="#444444",
        gridcolor="#E5E5E5",
        zeroline=False,
        title_standoff=8,
        exponentformat="power",
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family=FONTE, size=14, color="#222222"),
            title=dict(font=dict(size=16), x=0.0, xanchor="left"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=SEQUENCIA,
            xaxis=eixo,
            yaxis=eixo,
            legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=0),
            margin=dict(l=70, r=20, t=50, b=60),
            # Sem largura fixa no tema: no site a figura acompanha a largura da
            # página. A imagem do PDF usa LARGURA_PADRAO (ver mostrar()).
            height=ALTURA_PADRAO,
            hoverlabel=dict(font_family=FONTE),
            # Vírgula como separador decimal e ponto como separador de milhar.
            separators=",.",
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2.5, simplify=False), marker=dict(size=8))],
        ),
    )


def aplicar_estilo() -> None:
    """Registra o tema ``cnlivro`` do Plotly e o torna o tema padrão."""
    pio.templates["cnlivro"] = _template()
    pio.templates.default = "cnlivro"


def mostrar(fig: go.Figure, png: bool | None = None, escala: float = 2.0) -> None:
    """Mostra uma figura no notebook com versão interativa e, opcionalmente, estática.

    Parameters
    ----------
    fig:
        Figura do Plotly.
    png:
        Se ``True``, inclui também uma imagem PNG (necessária para o PDF).
        Se ``None``, segue a variável de ambiente ``CNLIVRO_PNG``.
    escala:
        Fator de resolução da imagem PNG.
    """
    from IPython.display import display

    if png is None:
        png = os.environ.get("CNLIVRO_PNG", "0") == "1"

    # Na web a figura ocupa a largura disponível da página (e se ajusta quando a
    # janela muda de tamanho), em vez de usar a largura fixa em pixels do tema
    # ou do capítulo. Isso evita a barra de rolagem horizontal em telas
    # estreitas. A largura fixa continua valendo para a imagem do PDF.
    web = go.Figure(fig)
    largura_original = fig.layout.width
    web.layout.width = None
    web.update_layout(autosize=True)
    html = web.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config=CONFIG_WEB,
        default_width="100%",
    )
    # Figuras largas (vários painéis lado a lado) ficam ilegíveis se espremidas
    # na tela de um celular: abaixo de uma largura mínima, elas rolam na
    # horizontal dentro do próprio quadro, sem alargar a página.
    if largura_original and largura_original > LARGURA_PADRAO:
        minimo = int(FRACAO_MINIMA * largura_original)
        html = (
            f'<div style="width:100%;overflow-x:auto">'
            f'<div style="min-width:{minimo}px">{html}</div></div>'
        )
    pacote = {
        "application/vnd.plotly.v1+json": {**web.to_plotly_json(), "config": CONFIG_WEB},
        "text/html": html,
    }
    if png:
        try:
            # No PDF, controles (sliders e botões) não fazem sentido: a imagem
            # mostra só o estado inicial da figura.
            estatica = go.Figure(fig)
            if fig.layout.sliders and fig.layout.height:
                estatica.update_layout(height=fig.layout.height - 90)  # espaço do slider
            estatica.layout.sliders = ()
            estatica.layout.updatemenus = ()
            largura = estatica.layout.width or LARGURA_PADRAO
            pacote["image/png"] = estatica.to_image(format="png", width=largura, scale=escala)
        except Exception as erro:  # kaleido/Chrome ausente: segue só com a versão web
            print(f"[cnlivro] PNG não gerado: {erro}")
    display(pacote, raw=True)


def com_controle(
    quadros,
    rotulo: str = "",
    ativo: int = 0,
    titulos=None,
) -> go.Figure:
    """Junta várias figuras numa só, com um slider para escolher qual mostrar.

    O controle é JavaScript puro do Plotly: funciona no site estático, sem
    Python rodando. No PDF, :func:`mostrar` desenha só o quadro ``ativo``.

    Parameters
    ----------
    quadros:
        Lista de pares ``(rótulo do passo, figura)``. Todas as figuras devem
        ter a mesma estrutura de eixos; o layout vem da primeira.
    rotulo:
        Texto mostrado antes do valor atual do slider (ex.: ``"x₀ = "``).
    ativo:
        Índice do quadro mostrado inicialmente.
    titulos:
        Lista opcional com o título da figura em cada quadro.
    """
    import copy

    fig = go.Figure(layout=copy.deepcopy(quadros[0][1].layout))
    faixas = []
    for i, (_, f) in enumerate(quadros):
        inicio = len(fig.data)
        for tr in f.data:
            tr = copy.deepcopy(tr)
            tr.visible = i == ativo
            fig.add_trace(tr)
        faixas.append((inicio, len(fig.data)))

    passos = []
    for i, (nome, _) in enumerate(quadros):
        visivel = [faixas[i][0] <= j < faixas[i][1] for j in range(len(fig.data))]
        args = [{"visible": visivel}]
        if titulos:
            args.append({"title.text": titulos[i]})
        passos.append(dict(method="update", label=nome, args=args))

    altura = (fig.layout.height or ALTURA_PADRAO) + 90
    fig.update_layout(
        sliders=[dict(active=ativo, steps=passos, currentvalue=dict(prefix=rotulo, font=dict(size=14)),
                      pad=dict(t=50, b=10), len=0.9, x=0.05)],
        height=altura,
    )
    if titulos:
        fig.update_layout(title=titulos[ativo])
    return fig


def numero(valor: float, casas: int | None = None) -> str:
    """Formata um número com vírgula decimal, para títulos e rótulos."""
    texto = f"{valor:.{casas}f}" if casas is not None else f"{valor:g}"
    return texto.replace(".", ",")


def reta_de_ordem(x_min: float, x_max: float, y_ref: float, ordem: float, estilo_traco: str = "dash"):
    """Traço de referência com inclinação ``ordem`` num gráfico log-log.

    A reta passa por ``(x_max, y_ref)``.
    """
    x = [x_min, x_max]
    y = [y_ref * (x_min / x_max) ** ordem, y_ref]
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=CORES["referencia"], dash=estilo_traco, width=1.5),
        showlegend=False,
        hoverinfo="skip",
    )


def grafico_convergencia(
    h,
    erro,
    ordens=(1, 2),
    nome: str = "erro medido",
    titulo: str = "Erro em função do passo",
    rotulo_x: str = "passo h",
    rotulo_y: str = "erro absoluto",
    deslocamento: float = 4.0,
) -> go.Figure:
    """Gráfico log-log do erro em função do passo, com as retas de ordem teórica.

    É o gráfico central da verificação de um método: se os pontos ficam
    paralelos à reta de ordem *p*, o método se comporta como a teoria prevê.
    As retas de referência ficam um pouco acima da curva (fator
    ``deslocamento``) para não a esconderem, e são rotuladas na ponta.
    """
    import numpy as np

    h = np.asarray(h, dtype=float)
    erro = np.asarray(erro, dtype=float)
    i_max = int(np.argmax(h))
    tracos = ["dash", "dot", "dashdot"]

    fig = go.Figure()
    for j, p in enumerate(ordens):
        reta = reta_de_ordem(h.min(), h.max(), erro[i_max] * deslocamento, p, tracos[j % 3])
        fig.add_trace(reta)
        fig.add_annotation(
            x=np.log10(h.min()), y=np.log10(reta.y[0]), text=f"ordem {p:g}",
            showarrow=False, xanchor="right", xshift=-4,
            font=dict(color=CORES["referencia"], size=12),
        )
    fig.add_trace(
        go.Scatter(
            x=h,
            y=erro,
            mode="lines+markers",
            name=nome,
            line=dict(color=CORES["erro"]),
            hovertemplate="h = %{x:.3e}<br>erro = %{y:.3e}<extra></extra>",
        )
    )
    fig.update_xaxes(type="log", title=rotulo_x)
    fig.update_yaxes(type="log", title=rotulo_y)
    fig.update_layout(title=titulo, margin=dict(l=110))
    return fig


def grafico_iteracoes(resultados: dict, titulo: str = "Erro em cada iteração") -> go.Figure:
    """Compara o erro aproximado por iteração de vários métodos (escala log em y).

    ``resultados`` é um dicionário ``{nome do método: Resultado}``, com os
    objetos devolvidos pelas funções de :mod:`cnlivro.raizes`.
    """
    fig = go.Figure()
    for nome, r in resultados.items():
        erros = [e for e in r.erros if e > 0]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(erros) + 1)),
                y=erros,
                mode="lines+markers",
                name=nome,
                hovertemplate="iteração %{x}<br>erro = %{y:.2e}<extra></extra>",
            )
        )
    fig.update_xaxes(title="iteração")
    fig.update_yaxes(type="log", title="erro relativo aproximado")
    fig.update_layout(title=titulo)
    return fig
