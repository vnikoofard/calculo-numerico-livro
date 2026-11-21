"""Sistemas de equações lineares A x = b (cap. 4).

Implementações de REFERÊNCIA, curtas e legíveis. Servem para comparar com o
código que o aluno pede ao agente, não para substituir ``numpy.linalg.solve``
ou ``scipy.sparse.linalg``.

Métodos diretos devolvem a solução (ou os fatores). Métodos iterativos devolvem
um :class:`ResultadoVetorial` com o histórico das iterações e dos resíduos,
que é o que permite verificar o método.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResultadoVetorial:
    """Saída comum aos métodos iterativos com incógnita vetorial (caps. 4 e 5).

    ``erros`` guarda a medida de convergência usada como critério de parada:
    o resíduo relativo ‖b − A x‖/‖b‖ nos sistemas lineares, ‖Δx‖/‖x‖ no método
    de Newton e ‖∇f‖ na otimização.
    """

    x: np.ndarray
    convergiu: bool
    iteracoes: int
    historico: list = field(default_factory=list)  # x_0, x_1, ..., x_n
    erros: list = field(default_factory=list)
    motivo: str = ""

    def __repr__(self) -> str:
        estado = "convergiu" if self.convergiu else f"NÃO convergiu ({self.motivo})"
        return f"ResultadoVetorial(x={np.array2string(np.asarray(self.x), precision=8)}, {estado}, iterações={self.iteracoes})"


# ---------------------------------------------------------------------------
# Métodos diretos
# ---------------------------------------------------------------------------

def substituicao_regressiva(U, b):
    """Resolve U x = b com U triangular superior."""
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def substituicao_progressiva(L, b, diagonal_unitaria=False):
    """Resolve L y = b com L triangular inferior."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        y[i] = s if diagonal_unitaria else s / L[i, i]
    return y


def lu(A):
    """Fatoração LU com pivotamento parcial: devolve P, L, U com P A = L U.

    L tem diagonal unitária. Lança ``ValueError`` se a matriz for singular
    (pivô nulo mesmo depois da troca de linhas).
    """
    U = np.array(A, dtype=float)
    n = U.shape[0]
    if U.shape != (n, n):
        raise ValueError("A precisa ser quadrada.")
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if U[p, k] == 0.0:
            raise ValueError(f"Matriz singular: coluna {k} sem pivô não nulo.")
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    if U[n - 1, n - 1] == 0.0:
        raise ValueError("Matriz singular: último pivô nulo.")
    P = np.eye(n)[perm]
    return P, L, U


def resolve_lu(P, L, U, b):
    """Resolve A x = b a partir de P A = L U (duas substituições)."""
    y = substituicao_progressiva(L, P @ np.asarray(b, dtype=float), diagonal_unitaria=True)
    return substituicao_regressiva(U, y)


def eliminacao_gauss(A, b, pivotamento=True):
    """Eliminação de Gauss com (ou sem) pivotamento parcial.

    ``pivotamento=False`` existe só para mostrar no livro por que ele é
    necessário: não use essa opção em código de verdade.
    """
    M = np.array(A, dtype=float)
    v = np.array(b, dtype=float)
    n = len(v)
    for k in range(n - 1):
        if pivotamento:
            p = k + int(np.argmax(np.abs(M[k:, k])))
            if p != k:
                M[[k, p]] = M[[p, k]]
                v[[k, p]] = v[[p, k]]
        if M[k, k] == 0.0:
            raise ValueError(f"Pivô nulo na coluna {k}.")
        for i in range(k + 1, n):
            fator = M[i, k] / M[k, k]
            M[i, k:] -= fator * M[k, k:]
            v[i] -= fator * v[k]
    return substituicao_regressiva(M, v)


# ---------------------------------------------------------------------------
# Métodos iterativos
# ---------------------------------------------------------------------------

def _residuo_relativo(A, x, b, norma_b):
    return float(np.linalg.norm(b - A @ x) / norma_b)


def _iterativo(passo, A, b, x0, tol, max_iter):
    A = np.asarray(A, dtype=float) if not hasattr(A, "tocsr") else A
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    norma_b = np.linalg.norm(b) or 1.0
    historico = [x.copy()]
    erros = [_residuo_relativo(A, x, b, norma_b)]
    for k in range(1, max_iter + 1):
        x = passo(x)
        historico.append(x.copy())
        erros.append(_residuo_relativo(A, x, b, norma_b))
        if not np.isfinite(erros[-1]):
            return ResultadoVetorial(x, False, k, historico, erros, "divergiu")
        if erros[-1] < tol:
            return ResultadoVetorial(x, True, k, historico, erros, "resíduo relativo < tol")
    return ResultadoVetorial(x, False, max_iter, historico, erros, "max_iter atingido")


def jacobi(A, b, x0=None, tol=1e-10, max_iter=10_000):
    """Método de Jacobi. Para pelo resíduo relativo ‖b − A x‖/‖b‖ < tol."""
    A_ = np.asarray(A, dtype=float)
    d = np.diag(A_)
    R = A_ - np.diag(d)
    b_ = np.asarray(b, dtype=float)
    return _iterativo(lambda x: (b_ - R @ x) / d, A_, b_, x0, tol, max_iter)


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10_000):
    """Método de Gauss-Seidel. Para pelo resíduo relativo ‖b − A x‖/‖b‖ < tol."""
    A_ = np.asarray(A, dtype=float)
    b_ = np.asarray(b, dtype=float)
    n = len(b_)

    def passo(x):
        x = x.copy()
        for i in range(n):
            x[i] = (b_[i] - A_[i, :i] @ x[:i] - A_[i, i + 1:] @ x[i + 1:]) / A_[i, i]
        return x

    return _iterativo(passo, A_, b_, x0, tol, max_iter)


def gradiente_conjugado(A, b, x0=None, tol=1e-10, max_iter=None):
    """Gradiente conjugado para A simétrica positiva definida (densa ou esparsa)."""
    b = np.asarray(b, dtype=float)
    n = len(b)
    max_iter = 10 * n if max_iter is None else max_iter
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    norma_b = np.linalg.norm(b) or 1.0
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    historico, erros = [x.copy()], [float(np.sqrt(rr) / norma_b)]
    if erros[0] < tol:
        return ResultadoVetorial(x, True, 0, historico, erros, "resíduo relativo < tol")
    for k in range(1, max_iter + 1):
        Ap = A @ p
        pAp = p @ Ap
        if pAp <= 0:
            return ResultadoVetorial(x, False, k, historico, erros, "A não é positiva definida")
        alfa = rr / pAp
        x = x + alfa * p
        r = r - alfa * Ap
        rr_novo = r @ r
        historico.append(x.copy())
        erros.append(float(np.sqrt(rr_novo) / norma_b))
        if erros[-1] < tol:
            return ResultadoVetorial(x, True, k, historico, erros, "resíduo relativo < tol")
        p = r + (rr_novo / rr) * p
        rr = rr_novo
    return ResultadoVetorial(x, False, max_iter, historico, erros, "max_iter atingido")


# ---------------------------------------------------------------------------
# Um problema-modelo: temperatura numa placa (Laplace, 5 pontos)
# ---------------------------------------------------------------------------

def placa_laplace(N, topo=100.0, base=0.0, esquerda=0.0, direita=0.0):
    """Monta o sistema esparso A T = b da placa quadrada com N × N nós internos.

    Cada nó interno satisfaz 4 T_ij − (soma dos 4 vizinhos) = 0, com as
    temperaturas das bordas passadas para o lado direito. A numeração é por
    linhas, de baixo para cima: o nó (i, j) vira o índice i·N + j, com i a
    linha (de baixo para cima) e j a coluna. Devolve A (``scipy.sparse.csr``) e b.
    """
    from scipy import sparse

    T = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N, N))
    I = sparse.identity(N)
    A = (sparse.kron(I, T) + sparse.kron(T, I)).tocsr()
    b = np.zeros((N, N))
    b[0, :] += base
    b[-1, :] += topo
    b[:, 0] += esquerda
    b[:, -1] += direita
    return A, b.ravel()
