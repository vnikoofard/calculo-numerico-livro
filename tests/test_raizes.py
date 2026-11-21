import math

import pytest

from cnlivro import raizes
from cnlivro.verificacao import ordem_iterativa, ordem_observada

RAIZ2 = math.sqrt(2)


def f(x):
    return x**2 - 2


def df(x):
    return 2 * x


@pytest.mark.parametrize(
    "resultado",
    [
        raizes.bissecao(f, 0, 2),
        raizes.falsa_posicao(f, 0, 2),
        raizes.newton(f, df, 1.0),
        raizes.secante(f, 1.0, 2.0),
        raizes.ponto_fixo(lambda x: x - (x**2 - 2) / 3, 1.0),
    ],
)
def test_todos_encontram_raiz_de_2(resultado):
    assert resultado.convergiu
    assert resultado.raiz == pytest.approx(RAIZ2, rel=1e-9)


def test_bissecao_exige_troca_de_sinal():
    with pytest.raises(ValueError):
        raizes.bissecao(f, 2, 3)


def test_newton_converge_quadraticamente():
    r = raizes.newton(f, df, 3.0)
    assert ordem_iterativa(r.historico, RAIZ2) == pytest.approx(2, abs=0.2)


def test_newton_em_raiz_multipla_e_linear():
    # (x - 1)^2 tem raiz dupla: Newton perde a convergência quadrática.
    r = raizes.newton(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 2.0, max_iter=100)
    assert ordem_iterativa(r.historico, 1.0) == pytest.approx(1, abs=0.1)


def test_secante_tem_ordem_aurea():
    r = raizes.secante(f, 3.0, 2.5)
    phi = (1 + math.sqrt(5)) / 2
    assert ordem_iterativa(r.historico, RAIZ2) == pytest.approx(phi, abs=0.3)


def test_newton_cicla_em_caso_patologico():
    # x^3 - 2x + 2 com x0 = 0 alterna entre 0 e 1 para sempre.
    r = raizes.newton(lambda x: x**3 - 2 * x + 2, lambda x: 3 * x**2 - 2, 0.0, max_iter=20)
    assert not r.convergiu
    assert r.historico[:4] == [0.0, 1.0, 0.0, 1.0]


def test_ordem_observada_diferenca_central():
    h = [0.1 / 2**k for k in range(6)]
    erro = [abs((math.sin(1 + hk) - math.sin(1 - hk)) / (2 * hk) - math.cos(1)) for hk in h]
    assert ordem_observada(h, erro) == pytest.approx(2, abs=0.05)
