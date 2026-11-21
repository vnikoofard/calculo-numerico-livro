import numpy as np
import pytest

from cnlivro.lineares import (eliminacao_gauss, gauss_seidel, gradiente_conjugado, jacobi, lu,
                              placa_laplace, resolve_lu, substituicao_progressiva,
                              substituicao_regressiva)

RNG = np.random.default_rng(0)


def test_substituicoes():
    U = np.triu(RNG.normal(size=(5, 5))) + 5 * np.eye(5)
    L = U.T
    b = RNG.normal(size=5)
    assert np.allclose(U @ substituicao_regressiva(U, b), b)
    assert np.allclose(L @ substituicao_progressiva(L, b), b)


@pytest.mark.parametrize("n", [2, 5, 20])
def test_lu_fatora_e_resolve(n):
    A = RNG.normal(size=(n, n))
    b = RNG.normal(size=n)
    P, L, U = lu(A)
    assert np.allclose(P @ A, L @ U)
    assert np.allclose(np.diag(L), 1.0)
    assert np.all(np.abs(L) <= 1.0 + 1e-14)          # pivotamento parcial: |l_ij| ≤ 1
    assert np.allclose(resolve_lu(P, L, U, b), np.linalg.solve(A, b))


def test_gauss_com_pivo_zero_na_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    assert np.allclose(eliminacao_gauss(A, b), [1.0, 1.0])
    with pytest.raises(ValueError):
        eliminacao_gauss(A, b, pivotamento=False)


def test_gauss_sem_pivotamento_perde_precisao():
    eps = 1e-17
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    assert np.allclose(eliminacao_gauss(A, b), [1.0, 1.0])
    assert not np.allclose(eliminacao_gauss(A, b, pivotamento=False), [1.0, 1.0])


def test_lu_singular():
    with pytest.raises(ValueError):
        lu(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_iterativos_na_placa():
    A, b = placa_laplace(6)
    x_ref = np.linalg.solve(A.toarray(), b)
    for metodo in (jacobi, gauss_seidel):
        r = metodo(A.toarray(), b, tol=1e-10)
        assert r.convergiu and np.allclose(r.x, x_ref, atol=1e-7)
    r = gradiente_conjugado(A, b, tol=1e-12)
    assert r.convergiu and np.allclose(r.x, x_ref)
    # Gauss-Seidel converge mais rápido que Jacobi; CG muito mais rápido que ambos
    assert gauss_seidel(A.toarray(), b, tol=1e-8).iteracoes < jacobi(A.toarray(), b, tol=1e-8).iteracoes
    assert r.iteracoes < 20


def test_placa_simetria_centro_25():
    N = 9                                           # nó central existe para N ímpar
    A, b = placa_laplace(N)
    T = np.linalg.solve(A.toarray(), b).reshape(N, N)
    assert T[N // 2, N // 2] == pytest.approx(25.0)


def test_cg_termina_em_n_passos():
    M = RNG.normal(size=(8, 8))
    A = M @ M.T + 8 * np.eye(8)
    b = RNG.normal(size=8)
    r = gradiente_conjugado(A, b, tol=1e-13)
    assert r.convergiu and r.iteracoes <= 8 + 1
