import numpy as np
import pytest

from cnlivro.derivacao import (centrada, derivada_dados, passo_complexo, progressiva, regressiva, richardson,
                               segunda_centrada)
from cnlivro.integracao import (gauss_legendre, monte_carlo, romberg, simpson, simpson_adaptativo, trapezio,
                                trapezio_dados)
from cnlivro.verificacao import ordem_observada


# ---- derivação --------------------------------------------------------------

def test_ordens_das_diferencas():
    hs = np.array([0.1, 0.05, 0.025, 0.0125])
    x = 0.7
    ex = np.cos(x)
    assert ordem_observada(hs, [abs(progressiva(np.sin, x, h) - ex) for h in hs]) == pytest.approx(1, abs=0.1)
    assert ordem_observada(hs, [abs(regressiva(np.sin, x, h) - ex) for h in hs]) == pytest.approx(1, abs=0.1)
    assert ordem_observada(hs, [abs(centrada(np.sin, x, h) - ex) for h in hs]) == pytest.approx(2, abs=0.1)
    assert ordem_observada(hs, [abs(segunda_centrada(np.sin, x, h) + np.sin(x)) for h in hs]) == pytest.approx(2, abs=0.1)


def test_richardson_e_passo_complexo():
    r = richardson(np.log, 2.0, 0.1, niveis=4)
    assert r.valor == pytest.approx(0.5, abs=1e-12)
    assert r.tabela[1, 1] == pytest.approx((4 * centrada(np.log, 2, 0.05) - centrada(np.log, 2, 0.1)) / 3)
    assert passo_complexo(np.exp, 1.0) == pytest.approx(np.e, rel=1e-15)


def test_derivada_dados_irregular():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 3, 40))
    y = np.sin(x)
    assert np.allclose(derivada_dados(x, y), np.gradient(y, x, edge_order=2))
    # exata para parábolas, inclusive nas pontas
    assert np.allclose(derivada_dados(x, 3 * x**2 - x + 2), 6 * x - 1)
    with pytest.raises(ValueError):
        derivada_dados(x[::-1], y)


# ---- integração -------------------------------------------------------------

def test_newton_cotes_e_gauss_exatidao():
    p = lambda x: 4 * x**3 - 3 * x**2 + 2 * x - 1        # ∫₀² = 16 − 8 + 4 − 2 = 10
    assert simpson(p, 0, 2, 2) == pytest.approx(10)
    assert gauss_legendre(p, 0, 2, 2) == pytest.approx(10)
    assert gauss_legendre(lambda x: x**9, 0, 1, 5) == pytest.approx(0.1)
    assert trapezio(lambda x: 2 * x + 1, 0, 3, 1) == pytest.approx(12)
    with pytest.raises(ValueError):
        simpson(p, 0, 2, 3)


def test_ordens_trapezio_simpson():
    ns = np.array([8, 16, 32, 64])
    ex = 1 - np.cos(1.0)
    e_t = [abs(trapezio(np.sin, 0, 1, n) - ex) for n in ns]
    e_s = [abs(simpson(np.sin, 0, 1, n) - ex) for n in ns]
    assert ordem_observada(1 / ns, e_t) == pytest.approx(2, abs=0.05)
    assert ordem_observada(1 / ns, e_s) == pytest.approx(4, abs=0.1)


def test_trapezio_dados():
    x = np.array([0.0, 0.3, 1.0, 1.5])
    assert trapezio_dados(x, 2 * x) == pytest.approx(1.5**2)


def test_romberg_e_adaptativo():
    r = romberg(np.exp, 0, 1, tol=1e-12)
    assert r.convergiu and r.valor == pytest.approx(np.e - 1, rel=1e-12)
    f = lambda x: 1 / (1e-4 + (x - 0.3) ** 2)
    ex = (np.arctan(0.7 / 1e-2) + np.arctan(0.3 / 1e-2)) / 1e-2
    a = simpson_adaptativo(f, 0, 1, tol=1e-8)
    assert a.valor == pytest.approx(ex, rel=1e-8)
    larguras = [d - c for c, d in a.intervalos]
    assert min(larguras) < max(larguras) / 100            # refina só perto do pico


def test_monte_carlo_erro_padrao():
    rng = np.random.default_rng(3)
    r = monte_carlo(lambda p: np.sum(p**2, axis=1), [(0, 1)] * 5, 200_000, rng)
    assert abs(r.valor - 5 / 3) < 4 * r.erros[0]
    assert r.erros[0] == pytest.approx(np.sqrt(5 * 4 / 45) / np.sqrt(200_000), rel=0.05)
