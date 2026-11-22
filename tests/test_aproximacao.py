import numpy as np
import pytest

from cnlivro.ajuste import equacoes_normais, gauss_newton, minimos_quadrados_qr, r2
from cnlivro.autovalores import metodo_potencia, predio_cisalhamento
from cnlivro.interpolacao import (diferencas_divididas, lagrange, newton_avalia, nos_chebyshev,
                                  spline_cubica_natural)

RNG = np.random.default_rng(0)


# ---- interpolação ----------------------------------------------------------

def test_newton_e_lagrange_reproduzem_polinomio():
    p = np.polynomial.Polynomial([1, -2, 0.5, 3])
    x = np.array([-1.0, 0.0, 0.7, 2.0])
    t = np.linspace(-1, 2, 50)
    c = diferencas_divididas(x, p(x))
    assert np.allclose(newton_avalia(c, x, t), p(t))
    assert np.allclose(lagrange(x, p(x), t), p(t))
    assert np.allclose(lagrange(x, p(x), x), p(x))          # nos próprios nós


def test_diferencas_divididas_a_mao():
    c = diferencas_divididas([20, 40, 60], [1.002, 0.653, 0.467])
    assert c == pytest.approx([1.002, -0.01745, 0.00020375])


def test_chebyshev_vence_runge():
    f = lambda x: 1 / (1 + 25 * x**2)
    t = np.linspace(-1, 1, 2001)
    xe = np.linspace(-1, 1, 21)
    xc = nos_chebyshev(21)
    erro_eq = np.abs(lagrange(xe, f(xe), t) - f(t)).max()
    erro_ch = np.abs(lagrange(xc, f(xc), t) - f(t)).max()
    assert erro_eq > 10 and erro_ch < 0.02


def test_spline_natural_contra_scipy_e_ordem():
    from scipy.interpolate import CubicSpline
    x = np.sort(RNG.uniform(0, 5, 9))
    y = np.sin(x)
    t = np.linspace(x[0], x[-1], 200)
    assert np.allclose(spline_cubica_natural(x, y)(t), CubicSpline(x, y, bc_type="natural")(t))
    with pytest.raises(ValueError):
        spline_cubica_natural(x[::-1], y[::-1])
    # erro longe das pontas cai como h⁴
    erros = []
    for n in (10, 20, 40):
        xn = np.linspace(0, np.pi, n + 1)
        tt = np.linspace(1.0, 2.0, 101)
        erros.append(np.abs(spline_cubica_natural(xn, np.sin(xn))(tt) - np.sin(tt)).max())
    assert np.log2(erros[1] / erros[2]) == pytest.approx(4, abs=0.5)


# ---- mínimos quadrados -----------------------------------------------------

def test_minimos_quadrados_qr_e_normais():
    A = RNG.normal(size=(30, 4))
    b = RNG.normal(size=30)
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(minimos_quadrados_qr(A, b), ref)
    assert np.allclose(equacoes_normais(A, b), ref)
    x = minimos_quadrados_qr(A, b)
    assert np.allclose(A.T @ (b - A @ x), 0, atol=1e-10)     # resíduo ⟂ colunas


def test_lauchli_normais_perdem_tudo():
    d = 1e-8
    A = np.array([[1, 1], [d, 0], [0, d]])
    b = np.array([2, d, d])
    assert np.allclose(minimos_quadrados_qr(A, b), [1, 1])
    with pytest.raises(np.linalg.LinAlgError):
        equacoes_normais(A, b)


def test_r2():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, y) == 1.0
    assert r2(y, np.full(3, 2.0)) == 0.0


def test_gauss_newton_exponencial():
    t = np.linspace(0, 10, 30)
    p_ref = np.array([80.0, 3.0])
    y = 20 + p_ref[0] * np.exp(-t / p_ref[1])
    res = lambda p: 20 + p[0] * np.exp(-t / p[1]) - y
    jac = lambda p: np.column_stack([np.exp(-t / p[1]), p[0] * t / p[1] ** 2 * np.exp(-t / p[1])])
    r = gauss_newton(res, jac, [60.0, 2.0])
    assert r.convergiu and np.allclose(r.x, p_ref)


# ---- autovalores -----------------------------------------------------------

def test_potencia_e_inversa():
    M = RNG.normal(size=(6, 6))
    A = M + M.T
    lams = np.linalg.eigvalsh(A)
    r = metodo_potencia(A, x0=RNG.normal(size=6), tol=1e-13, max_iter=100_000)
    assert r.historico_autovalor[-1] == pytest.approx(lams[np.argmax(np.abs(lams))], rel=1e-6)
    alvo = lams[2] + 0.1
    r = metodo_potencia(A, x0=RNG.normal(size=6), inversa=True, deslocamento=alvo, tol=1e-14)
    assert r.historico_autovalor[-1] == pytest.approx(lams[2], rel=1e-8)


def test_predio_cisalhamento():
    M, K = predio_cisalhamento([1.0, 1.0], [2.0, 1.0])
    assert np.allclose(K, [[3, -1], [-1, 1]])
    assert np.allclose(M, np.eye(2))


# ---- B-splines --------------------------------------------------------------

def test_base_bspline_contra_scipy_e_particao_da_unidade():
    from scipy.interpolate import BSpline
    from cnlivro.interpolacao import base_bspline, nos_bspline
    x = np.linspace(0, 2, 301)                          # inclui as duas pontas
    for k in (0, 1, 2, 3):
        t = nos_bspline(0, 2, 5, k)
        B = base_bspline(x, t, k)
        assert B.shape == (len(x), 5 + k)
        assert np.allclose(B.sum(axis=1), 1)            # partição da unidade
        assert np.all(B >= 0)
        ref = BSpline.design_matrix(x, t, k).toarray()
        assert np.allclose(B, ref)
    # suporte local: B_{i,k} é zero fora de [t_i, t_{i+k+1}]
    t = nos_bspline(0, 1, 6, 3)
    B = base_bspline(x / 2, t, 3)
    for i in range(B.shape[1]):
        fora = (x / 2 < t[i]) | (x / 2 > t[i + 4])
        assert np.all(B[fora, i] == 0)


def test_base_bspline_em_pytorch_e_integral():
    torch = pytest.importorskip("torch")
    from cnlivro.interpolacao import base_bspline, nos_bspline
    t = nos_bspline(-1, 1, 4, 3)
    x = np.linspace(-1, 1, 41)
    Bt = base_bspline(torch.tensor(x, dtype=torch.float64), t, 3)
    assert isinstance(Bt, torch.Tensor)
    assert np.allclose(Bt.numpy(), base_bspline(x, t, 3))
    # ∫ B_{i,k} dx = (t_{i+k+1} - t_i) / (k + 1)
    xf = np.linspace(-1, 1, 20001)
    integrais = np.trapezoid(base_bspline(xf, t, 3), xf, axis=0)
    assert np.allclose(integrais, (t[4:] - t[:-4]) / 4, atol=1e-6)
