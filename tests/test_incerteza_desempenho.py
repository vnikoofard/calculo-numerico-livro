import numpy as np
import pytest
from scipy.linalg import solve_banded

from cnlivro.desempenho import cronometrar, dispositivo, thomas, thomas_lote
from cnlivro.incerteza import gradiente_torch, monte_carlo, probabilidade, propagacao_linear


def test_monte_carlo_e_erro_padrao():
    rng = np.random.default_rng(0)
    N = 200_000
    r = monte_carlo(lambda x, y: x + 2 * y, dict(x=rng.normal(1, 1, N), y=rng.normal(0, 0.5, N)))
    assert r.media == pytest.approx(1, abs=4 * r.erro_padrao)
    assert r.desvio == pytest.approx(np.sqrt(2), rel=0.01)
    assert r.erro_padrao == pytest.approx(np.sqrt(2 / N), rel=0.01)
    assert len(r.historico) == 5                   # médias com 10, 100, ..., 10⁵ amostras


def test_probabilidade_wilson():
    p, lo, hi = probabilidade(np.r_[np.ones(10, bool), np.zeros(90, bool)])
    assert p == 0.1 and lo < 0.1 < hi and hi - lo < 0.15
    p, lo, hi = probabilidade(np.zeros(1000, bool))     # nenhum evento: o limite superior não é zero
    assert p == 0 and lo < 1e-12 and 0.003 < hi < 0.005    # "regra do 3": cerca de 3/N
    # cobertura: o intervalo de 95 % contém o p verdadeiro em ~95 % das repetições
    rng = np.random.default_rng(1)
    cobre = [lo <= 0.05 <= hi for lo, hi in (probabilidade(rng.random(400) < 0.05)[1:] for _ in range(2000))]
    assert 0.93 < np.mean(cobre) < 0.97


def test_propagacao_linear_e_torch():
    f = lambda a, b: a**2 * np.exp(b)
    y, s, g = propagacao_linear(f, [2.0, 0.5], [0.1, 0.2])
    assert np.allclose(g, [4 * np.exp(0.5), 4 * np.exp(0.5)], rtol=1e-7)
    assert s == pytest.approx(np.sqrt((g[0] * 0.1) ** 2 + (g[1] * 0.2) ** 2))
    # linear: o método delta é exato
    rng = np.random.default_rng(2)
    _, s_lin, _ = propagacao_linear(lambda a, b: 3 * a - b, [1.0, 2.0], [0.2, 0.3])
    mc = monte_carlo(lambda a, b: 3 * a - b, dict(a=rng.normal(1, 0.2, 100_000), b=rng.normal(2, 0.3, 100_000)))
    assert s_lin == pytest.approx(mc.desvio, rel=0.01)
    import torch
    y_t, g_t = gradiente_torch(lambda a, b: a**2 * torch.exp(b), [2.0, 0.5])
    assert y_t == pytest.approx(y) and np.allclose(g_t, g, rtol=1e-7)


def test_thomas():
    rng = np.random.default_rng(3)
    n, lote = 50, 7
    a, c = rng.random((lote, n)), rng.random((lote, n))
    b = 3 + rng.random((lote, n))
    d = rng.random((lote, n))
    x = thomas_lote(a, b, c, d)
    for k in range(lote):
        ab = np.vstack([np.r_[0, c[k, :-1]], b[k], np.r_[a[k, 1:], 0]])
        assert np.allclose(x[k], solve_banded((1, 1), ab, d[k]))
    assert np.allclose(thomas(a[0], b[0], c[0], d[0]), x[0])


def test_cronometrar():
    chamadas = []
    r = cronometrar(lambda v: chamadas.append(v) or v * 2, 21, repeticoes=4, aquecimento=2)
    assert len(chamadas) == 6 and len(r.tempos) == 4 and r.valor == 42
    assert 0 <= r.minimo <= r.mediana
    assert dispositivo() in ("cuda", "mps", "cpu")
