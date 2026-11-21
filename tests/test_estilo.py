import plotly.graph_objects as go

from cnlivro.estilo import com_controle, numero


def test_numero_usa_virgula():
    assert numero(0.5) == "0,5"
    assert numero(1.23456, 2) == "1,23"


def test_com_controle_mostra_so_o_quadro_ativo():
    quadros = [(str(i), go.Figure([go.Scatter(y=[i]), go.Scatter(y=[2 * i])])) for i in range(3)]
    fig = com_controle(quadros, rotulo="k = ", ativo=1, titulos=["a", "b", "c"])
    assert [t.visible for t in fig.data] == [False, False, True, True, False, False]
    passos = fig.layout.sliders[0].steps
    assert len(passos) == 3
    assert passos[2].args[0]["visible"] == [False, False, False, False, True, True]
    assert fig.layout.title.text == "b"
