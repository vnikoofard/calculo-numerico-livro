import numpy as np
import pytest

from cnlivro.edo import (adams_bashforth2, autoestados, contorno_dif_finitas, implicito, passo_fixo, rk45, tiro,
                         verlet)
from cnlivro.verificacao import ordem_observada


def f_teste(t, y):
    return -2 * t * y            # y = exp(-t²)


def test_ordens_passo_fixo():
    hs = np.array([0.1, 0.05, 0.025, 0.0125])
    ex = np.exp(-1.0)
    for metodo, p in [("euler", 1), ("heun", 2), ("rk4", 4)]:
        erros = [abs(passo_fixo(f_teste, (0, 1), 1.0, h, metodo).y[-1, 0] - ex) for h in hs]
        assert ordem_observada(hs, erros) == pytest.approx(p, abs=0.15)
    r = passo_fixo(f_teste, (0, 1), 1.0, 0.3, "rk4")
    assert r.t[-1] == pytest.approx(1.0) and r.avaliacoes == 16


def test_rk45_tolerancia_e_scipy():
    from scipy.integrate import solve_ivp
    for rtol in [1e-4, 1e-8]:
        r = rk45(f_teste, (0, 3), [1.0], rtol=rtol, atol=1e-12)
        assert r.sucesso
        assert abs(r.y[-1, 0] - np.exp(-9)) < 1e3 * rtol * 1e-3 + 1e-8
    # mesmo método que o solve_ivp: número de passos parecido
    r = rk45(f_teste, (0, 3), [1.0], rtol=1e-6, atol=1e-9)
    s = solve_ivp(f_teste, (0, 3), [1.0], rtol=1e-6, atol=1e-9)
    assert abs(len(r.t) - len(s.t)) <= 0.3 * len(s.t)


def test_implicitos():
    lam = -1000.0
    f = lambda t, y: lam * (y - np.cos(t))
    # Euler implícito é estável com h muito acima de 2/|λ|
    r = implicito(f, (0, 2), [0.0], 0.1, "euler", jac=lambda t, y: np.array([[lam]]))
    assert r.sucesso and np.all(np.abs(r.y) < 2)
    assert max(r.historico) <= 2               # linear: Newton converge em 1 a 2 iterações
    # ordens
    g = lambda t, y: -y + np.sin(t)
    ex = 0.5 * (np.sin(1) - np.cos(1)) + 1.5 * np.exp(-1)
    hs = np.array([0.1, 0.05, 0.025])
    for metodo, p in [("euler", 1), ("trapezio", 2)]:
        erros = [abs(implicito(g, (0, 1), [1.0], h, metodo).y[-1, 0] - ex) for h in hs]
        assert ordem_observada(hs, erros) == pytest.approx(p, abs=0.15)


def test_ab2_e_verlet():
    hs = np.array([0.02, 0.01, 0.005])
    erros = [abs(adams_bashforth2(f_teste, (0, 1), 1.0, h).y[-1, 0] - np.exp(-1)) for h in hs]
    assert ordem_observada(hs, erros) == pytest.approx(2, abs=0.15)
    # oscilador: Verlet conserva a energia sem deriva em tempos longos
    r = verlet(lambda q: -q, (0, 1000), [1.0], [0.0], 0.1)
    E = 0.5 * (r.y[:, 0] ** 2 + r.y[:, 1] ** 2)
    assert np.max(np.abs(E - 0.5)) < 2e-3
    assert r.avaliacoes == r.t.size


def test_contorno():
    # y'' = y, y(0) = 0, y(1) = sinh(1): solução sinh(x)
    for n in [20, 40]:
        r = contorno_dif_finitas(lambda x, y, dy: y, (0, 1), 0.0, ("dirichlet", np.sinh(1)), n)
        assert r.sucesso and len(r.historico) <= 2
    e = [np.max(np.abs(contorno_dif_finitas(lambda x, y, dy: y, (0, 1), 0.0, ("dirichlet", np.sinh(1)), n).y
                       - np.sinh(np.linspace(0, 1, n + 1)))) for n in [20, 40, 80]]
    assert ordem_observada(1 / np.array([20, 40, 80]), e) == pytest.approx(2, abs=0.1)
    # Neumann na ponta: aleta com ponta isolada, θ = cosh(m(L-x))/cosh(mL)
    m = 3.0
    r = contorno_dif_finitas(lambda x, y, dy: m**2 * y, (0, 1), 1.0, ("neumann", 0.0), 200)
    assert np.max(np.abs(r.y - np.cosh(m * (1 - r.t)) / np.cosh(m))) < 1e-4
    # não linear (Bratu): y'' = -e^y, y(0) = y(1) = 0, solução inferior com y(1/2) ≈ 0,1405
    r = contorno_dif_finitas(lambda x, y, dy: -np.exp(y), (0, 1), 0.0, ("dirichlet", 0.0), 100)
    assert r.sucesso and r.y[50] == pytest.approx(0.140539, abs=1e-4)
    # tiro no mesmo problema linear
    t = tiro(lambda x, y, dy: y, (0, 1), 0.0, lambda yb, vb: yb - np.sinh(1), 0.0, 2.0)
    assert t.sucesso and t.historico[-1][0] == pytest.approx(1.0, abs=1e-8)


def test_autoestados_oscilador():
    E, x, psi = autoestados(lambda x: 0.5 * x**2, (-8, 8), 800, k=4)
    assert np.allclose(E, [0.5, 1.5, 2.5, 3.5], atol=1e-3)
    assert np.trapezoid(psi[:, 0] ** 2, x) == pytest.approx(1, abs=1e-6)
    # -ψ'' + x²ψ = Eψ: níveis 1, 3, 5, 7
    E, _, _ = autoestados(lambda x: x**2, (-8, 8), 800, k=4, coef=1.0)
    assert np.allclose(E, [1, 3, 5, 7], atol=2e-3)


def test_contorno_sem_solucao():
    # Frank-Kamenetskii acima do valor crítico (δ ≈ 0,878): não há solução, e o Newton deve falhar avisando
    r = contorno_dif_finitas(lambda x, y, dy: -0.95 * np.exp(y), (-1, 1), 0.0, ("dirichlet", 0.0), 100)
    assert not r.sucesso
