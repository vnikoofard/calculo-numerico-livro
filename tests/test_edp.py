import numpy as np
import pytest

from cnlivro.edp import (adveccao_difusao, burgers, calor_theta, fem_1d, laplaciano_1d, laplaciano_2d,
                         passo_maximo, poisson_2d)
from cnlivro.verificacao import ordem_observada


def gaussiana(x, t, u=1.0, D=0.0, x0=0.3, s0=0.05):
    s2 = s0**2 + 2 * D * t
    return s0 / np.sqrt(s2) * np.exp(-(x - x0 - u * t) ** 2 / (2 * s2))


def test_adveccao_ordens_e_courant():
    erros = {"upwind": [], "lax-wendroff": []}
    ns = np.array([200, 400, 800])
    for esquema in erros:
        for n in ns:
            x = np.linspace(0, 1, n + 1)
            r = adveccao_difusao(gaussiana(x, 0), x, 1.0, 0.0, 0.4, dt=0.5 / n, esquema=esquema)
            assert r.sucesso and r.historico[0]["C"] == pytest.approx(0.5)
            erros[esquema].append(np.max(np.abs(r.u[-1] - gaussiana(x, 0.4))))
    assert ordem_observada(1 / ns, erros["upwind"]) == pytest.approx(1, abs=0.2)
    assert ordem_observada(1 / ns, erros["lax-wendroff"]) == pytest.approx(2, abs=0.2)
    # Lax-Wendroff com difusão continua de ordem 2 (termos cruzados de Taylor)
    e = []
    for n in ns:
        x = np.linspace(0, 1, n + 1)[:-1]
        r = adveccao_difusao(gaussiana(x, 0, D=2e-4), x, 1.0, 2e-4, 0.4, dt=0.5 / n, esquema="lax-wendroff",
                             periodico=True)
        e.append(np.max(np.abs(r.u[-1] - gaussiana(x, 0.4, D=2e-4))))
    assert ordem_observada(1 / ns, e) == pytest.approx(2, abs=0.15)
    # upwind sem difusão com C = 1 transporta exatamente (periódico)
    x = np.linspace(0, 1, 101)[:-1]
    pulso = np.where(np.abs(x - 0.3) < 0.1, 1.0, 0.0)
    r = adveccao_difusao(pulso, x, 1.0, 0.0, 1.0, dt=0.01, periodico=True)
    assert np.allclose(r.u[-1], pulso)
    # C > 1: explode
    r = adveccao_difusao(pulso, x, 1.0, 0.0, 5.0, dt=0.012, periodico=True)
    assert not r.sucesso or np.max(np.abs(r.u[-1])) > 1e3


def test_passo_maximo():
    dx, u, D = 0.1, 2.0, 0.05
    for esquema in ["upwind", "lax-wendroff", "centrado"]:
        dt = passo_maximo(dx, u, D, esquema)
        C, r = u * dt / dx, D * dt / dx**2
        lim = {"upwind": C + 2 * r, "lax-wendroff": C**2 + 2 * r, "centrado": max(2 * r, C**2 / (2 * r))}[esquema]
        assert lim == pytest.approx(1)
    assert passo_maximo(dx, u, 0.0, "centrado") == 0.0


def test_calor_theta():
    exata = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
    ns = np.array([20, 40, 80])
    for theta in [0.0, 0.5, 1.0]:
        e = []
        for n in ns:
            x = np.linspace(0, 1, n + 1)
            dt = 0.4 / n**2 if theta == 0 else (1 / n if theta == 0.5 else 1 / n**2)
            e.append(np.max(np.abs(calor_theta(np.sin(np.pi * x), x, 1.0, 0.1, dt, theta).u[-1] - exata(x, 0.1))))
        assert ordem_observada(1 / ns, e) == pytest.approx(2, abs=0.15)
    # ordem no tempo: Crank-Nicolson 2, BTCS 1
    x = np.linspace(0, 1, 401)
    dts = np.array([0.02, 0.01, 0.005])
    for theta, p in [(0.5, 2), (1.0, 1)]:
        e = [np.max(np.abs(calor_theta(np.sin(np.pi * x), x, 1.0, 0.1, dt, theta).u[-1] - exata(x, 0.1))) for dt in dts]
        assert ordem_observada(dts, e) == pytest.approx(p, abs=0.15)
    # FTCS com r = 0,6 amplifica o ruído de arredondamento
    x = np.linspace(0, 1, 41)
    r = calor_theta(np.sin(np.pi * x), x, 1.0, 0.1, 0.6 / 40**2, 0.0)
    assert np.max(np.abs(r.u[-1])) > 1e6
    # Neumann (nó fantasma) e fonte
    x = np.linspace(0, 1, 81)
    r = calor_theta(np.cos(np.pi * x / 2), x, 1.0, 0.2, 1e-3, 0.5, esquerda=("neumann", 0.0))
    assert np.max(np.abs(r.u[-1] - np.exp(-np.pi**2 / 4 * 0.2) * np.cos(np.pi * x / 2))) < 2e-5
    r = calor_theta(np.zeros(81), x, 1.0, 20, 0.05, 1.0, esquerda=("dirichlet", 1.0), direita=("neumann", 2.0))
    assert np.allclose(r.u[-1], 1 + 2 * x, atol=1e-8)
    r = calor_theta(np.zeros(81), x, 1.0, 5, 0.01, 0.5, fonte=lambda x, t: 2 + 0 * x)
    assert np.allclose(r.u[-1], x * (1 - x), atol=1e-8)


def test_poisson_2d():
    exata = lambda X, Y: np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
    iteracoes = {}
    for metodo in ["direto", "jacobi", "gauss-seidel", "sor"]:
        r = poisson_2d(0, exata, 1, 1, 40, 40, metodo, tol=1e-10)
        X, Y = np.meshgrid(*r.x, indexing="ij")
        assert r.sucesso and np.max(np.abs(r.u - exata(X, Y))) < 2e-4
        iteracoes[metodo] = r.passos
    # Gauss-Seidel ~ metade das iterações de Jacobi; SOR ótimo, muito menos
    assert 0.4 < iteracoes["gauss-seidel"] / iteracoes["jacobi"] < 0.6
    assert iteracoes["sor"] < 0.1 * iteracoes["gauss-seidel"]
    # solução polinomial: exata (a menos do arredondamento), com nx != ny
    f = lambda X, Y: 2 * (Y * (1 - Y) + X * (1 - X))
    for metodo in ["direto", "sor"]:
        r = poisson_2d(f, 0, 1, 1, 20, 30, metodo, tol=1e-13)
        X, Y = np.meshgrid(*r.x, indexing="ij")
        assert r.u.shape == (21, 31) and np.max(np.abs(r.u - X * (1 - X) * Y * (1 - Y))) < 1e-11


def test_laplacianos():
    n, dx = 50, 0.02
    xc = (np.arange(n) + 0.5) * dx
    # Neumann (volumes finitos): soma das linhas zero, conserva a média
    L = laplaciano_1d(n, dx, "neumann")
    assert np.allclose(L @ np.ones(n), 0)
    assert np.allclose((L @ np.cos(np.pi * xc))[5:-5], -np.pi**2 * np.cos(np.pi * xc)[5:-5], rtol=2e-3)
    L2 = laplaciano_2d(n, n, dx, dx, "neumann")
    assert L2.shape == (n * n, n * n) and np.allclose(L2 @ np.ones(n * n), 0)
    Ld = laplaciano_1d(4, 1.0)
    assert np.allclose(Ld.toarray()[0], [-2, 1, 0, 0])


def test_burgers_velocidade_do_choque():
    x = np.linspace(-1, 3, 401)
    for uR in [0.0, 0.5]:
        u0 = np.where(x < 0, 1.0, uR)
        cons = burgers(u0, x, 2.0, 0.005, "conservativa")
        nao = burgers(u0, x, 2.0, 0.005, "nao-conservativa")
        frente = lambda v: x[np.argmax(v < (1 + uR) / 2)]
        assert frente(cons.u[-1]) == pytest.approx((1 + uR), abs=0.03)        # velocidade (1 + uR)/2
        assert abs(frente(nao.u[-1]) - (1 + uR)) > 0.05                        # velocidade errada
    # rarefação: u = x/t no leque
    u0 = np.where(x < 0, 0.0, 1.0)
    r = burgers(u0, x, 1.0, 0.005, "conservativa")
    meio = (x > 0.2) & (x < 0.8)
    assert np.max(np.abs(r.u[-1][meio] - x[meio])) < 0.06


def test_fem_1d():
    # exato nos nós para k constante, mesmo com poucos elementos
    nos = np.linspace(0, 1, 6)
    r = fem_1d(1.0, np.exp, nos, 0.0, ("dirichlet", 0.0))
    assert np.max(np.abs(r.u - (1 - np.exp(nos) + (np.e - 1) * nos))) < 1e-9
    # parede composta: a interface é tratada naturalmente
    nos = np.linspace(0, 1, 5)
    r = fem_1d(lambda x: np.where(x < 0.5, 1.0, 10.0), 0.0, nos, 1.0, ("dirichlet", 0.0))
    assert r.u[2] == pytest.approx(1 - 0.5 / 0.55)
    # Neumann natural: k u'(1) = q
    r = fem_1d(2.0, 0.0, np.linspace(0, 1, 4), 0.0, ("neumann", 4.0))
    assert np.allclose(r.u, 2 * np.linspace(0, 1, 4))
    K, F = r.historico
    assert np.allclose(K @ np.ones(4), 0)          # rigidez sem contorno: modo constante no núcleo
