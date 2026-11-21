import numpy as np
import pytest

from cnlivro.otimizacao import (bfgs, gradiente_descendente, grad_rosenbrock, newton_sistema,
                                rosenbrock)
from cnlivro.verificacao import ordem_iterativa


def F(v):
    x, y = v
    return np.array([x**2 + y**2 - 4.0, x * y - 1.0])


def J(v):
    x, y = v
    return np.array([[2 * x, 2 * y], [y, x]])


RAIZ = np.array([np.sqrt(2 + np.sqrt(3)), np.sqrt(2 - np.sqrt(3))])


def test_newton_sistema_converge_quadraticamente():
    r = newton_sistema(F, J, [2.0, 0.5], tol=1e-14)
    assert r.convergiu
    assert np.allclose(r.x, RAIZ)
    erros = [np.linalg.norm(x - RAIZ) for x in r.historico]
    erros = [e for e in erros if e > 1e-13]
    p = np.polyfit(np.log(erros[:-1]), np.log(erros[1:]), 1)[0]
    assert p == pytest.approx(2.0, abs=0.3)


def test_newton_sistema_jacobiana_singular():
    r = newton_sistema(F, J, [0.0, 0.0])
    assert not r.convergiu and "singular" in r.motivo


def test_gradiente_quadratica_taxa():
    kappa = 10.0
    Q = np.diag([1.0, kappa])
    f = lambda x: 0.5 * x @ Q @ x
    g = lambda x: Q @ x
    alfa = 2.0 / (1.0 + kappa)                     # passo ótimo fixo
    r = gradiente_descendente(f, g, [kappa, 1.0], passo=alfa, tol=1e-10)
    assert r.convergiu
    taxa = np.exp(np.mean(np.diff(np.log(r.erros[:30]))))
    assert taxa == pytest.approx((kappa - 1) / (kappa + 1), rel=1e-6)


def test_bfgs_rosenbrock():
    r = bfgs(rosenbrock, grad_rosenbrock, [-1.2, 1.0], tol=1e-8)
    assert r.convergiu
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-6)
    assert r.iteracoes < 100
    lento = gradiente_descendente(rosenbrock, grad_rosenbrock, [-1.2, 1.0], tol=1e-8, max_iter=2000)
    assert not lento.convergiu                      # o gradiente fica preso no vale
