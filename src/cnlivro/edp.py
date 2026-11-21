"""Equações diferenciais parciais (caps. 15 e 16).

Implementações de REFERÊNCIA, curtas e legíveis:

- advecção-difusão 1D explícita: upwind, Lax-Wendroff e centrada (:func:`adveccao_difusao`);
- calor 1D pelo método θ (FTCS, BTCS, Crank-Nicolson), com solução tridiagonal (:func:`calor_theta`);
- Poisson 2D: solução direta esparsa, Jacobi, Gauss-Seidel e SOR (:func:`poisson_2d`);
- Burgers 1D em forma conservativa e não conservativa (:func:`burgers`);
- laplacianos esparsos para o método das linhas (:func:`laplaciano_1d`, :func:`laplaciano_2d`);
- elementos finitos lineares 1D para -(k u')' = f (:func:`fem_1d`).

As funções devolvem :class:`ResultadoEDP`. Para uso em produção: ``scipy.sparse``,
``scipy.integrate.solve_ivp`` (método das linhas) e bibliotecas como py-pde,
FEniCSx ou scikit-fem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve


@dataclass
class ResultadoEDP:
    x: np.ndarray                     # malha (1D) ou tupla (x, y) (2D)
    t: np.ndarray                     # instantes guardados (vazio em problemas estacionários)
    u: np.ndarray                     # u[k] é a solução em t[k]; em problemas estacionários, a solução
    passos: int                       # passos de tempo ou iterações
    sucesso: bool = True
    motivo: str = ""
    historico: list = field(default_factory=list)  # parâmetros do esquema, ou variação por iteração

    def __repr__(self) -> str:
        estado = "sucesso" if self.sucesso else f"FALHOU ({self.motivo})"
        return f"ResultadoEDP({estado}, passos={self.passos})"


def _instantes_guardados(instantes, t_final, dt, n):
    """Índices dos passos em que a solução é guardada (sempre inclui o início e o fim)."""
    if instantes is None:
        return {0, n}
    return {0, n} | {min(n, int(round(t / dt))) for t in instantes}


# ---- advecção-difusão 1D ----------------------------------------------------

def passo_maximo(dx: float, u: float, D: float, esquema: str) -> float:
    """Maior passo de tempo estável (análise de von Neumann) para advecção-difusão explícita.

    upwind: C + 2r <= 1;  Lax-Wendroff: C² + 2r <= 1;  centrado: C² <= 2r <= 1,
    com C = |u| dt/dx (número de Courant) e r = D dt/dx².
    """
    a, d = abs(u) / dx, D / dx**2
    if esquema == "upwind":
        return 1.0 / (a + 2 * d)
    if esquema == "lax-wendroff":
        # C² + 2r = a² dt² + 2 d dt = 1
        if a == 0:
            return 1 / (2 * d)
        return (-d + np.sqrt(d**2 + a**2)) / a**2
    if esquema == "centrado":
        if d == 0:
            return 0.0                # sem difusão, o esquema centrado é instável para todo dt
        return min(1 / (2 * d), 2 * d / a**2) if a else 1 / (2 * d)
    raise ValueError(f"esquema desconhecido: {esquema}")


def adveccao_difusao(c0, x, u: float, D: float, t_final: float, dt: float | None = None,
                     esquema: str = "upwind", periodico: bool = False, instantes=None,
                     fator: float = 0.9) -> ResultadoEDP:
    """Integra c_t + u c_x = D c_xx numa malha uniforme ``x`` até ``t_final``.

    ``esquema``: "upwind" (ordem 1), "lax-wendroff" (ordem 2) ou "centrado" (FTCS).
    Com difusão, Lax-Wendroff inclui os termos cruzados da expansão de Taylor no tempo,
    -C r δ³c e (r²/2) δ⁴c, que mantêm a ordem 2.
    Sem ``dt``, usa ``fator`` vezes o passo máximo estável (:func:`passo_maximo`).
    Com ``periodico=False`` os valores das pontas ficam fixos (Dirichlet).
    ``instantes``: tempos em que a solução é guardada, além do início e do fim.
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    c = np.array(c0, dtype=float)
    if dt is None:
        dt = fator * passo_maximo(dx, u, D, esquema)
        if dt == 0:
            raise ValueError("o esquema centrado sem difusão é instável para qualquer passo: informe dt")
    n = int(np.ceil(t_final / dt - 1e-9))
    dt = t_final / n
    C, r = u * dt / dx, D * dt / dx**2
    guardar = _instantes_guardados(instantes, t_final, dt, n)
    ts, us = [0.0], [c.copy()]
    pontas = c[[0, -1]].copy()
    for k in range(1, n + 1):
        cm, cp = np.roll(c, 1), np.roll(c, -1)       # c_{i-1} e c_{i+1}
        dif2 = cp - 2 * c + cm
        if esquema == "upwind":
            adv = -C * (c - cm) if u >= 0 else -C * (cp - c)
        elif esquema == "lax-wendroff":
            # Taylor até Δt²/2 c_tt, com c_tt = u² c_xx - 2uD c_xxx + D² c_xxxx (ordem 2 também com difusão)
            adv = -C / 2 * (cp - cm) + C**2 / 2 * dif2
            if D:
                cmm, cpp = np.roll(c, 2), np.roll(c, -2)
                d3 = (cpp - 2 * cp + 2 * cm - cmm) / 2
                d4 = cpp - 4 * cp + 6 * c - 4 * cm + cmm
                if not periodico:          # perto das pontas, sem os vizinhos de distância 2
                    d3[[1, -2]] = d4[[1, -2]] = 0.0
                adv += -C * r * d3 + r**2 / 2 * d4
        elif esquema == "centrado":
            adv = -C / 2 * (cp - cm)
        else:
            raise ValueError(f"esquema desconhecido: {esquema}")
        c = c + adv + r * dif2
        if not periodico:
            c[[0, -1]] = pontas
        if not np.all(np.isfinite(c)) or np.max(np.abs(c)) > 1e100:
            ts.append(k * dt)
            us.append(c.copy())
            return ResultadoEDP(x, np.array(ts), np.array(us), k, False, "a solução explodiu (instabilidade)",
                                [dict(C=C, r=r, dt=dt)])
        if k in guardar:
            ts.append(k * dt)
            us.append(c.copy())
    return ResultadoEDP(x, np.array(ts), np.array(us), n, True, "", [dict(C=C, r=r, dt=dt)])


# ---- calor 1D pelo método θ -------------------------------------------------

def calor_theta(u0, x, alpha: float, t_final: float, dt: float, theta: float = 0.5,
                esquerda=("dirichlet", 0.0), direita=("dirichlet", 0.0),
                fonte: Callable | None = None, instantes=None) -> ResultadoEDP:
    """Integra u_t = alpha u_xx + fonte(x, t) numa malha uniforme ``x``.

    ``theta`` = 0: FTCS (explícito, estável para r = alpha dt/dx² <= 1/2);
    ``theta`` = 1: BTCS (Euler implícito); ``theta`` = 1/2: Crank-Nicolson.
    Cada ponta é ("dirichlet", valor) ou ("neumann", derivada), esta com nó fantasma.
    O sistema tridiagonal de cada passo é resolvido com ``solve_banded``.
    """
    x = np.asarray(x, dtype=float)
    N, dx = x.size, x[1] - x[0]
    u = np.array(u0, dtype=float)
    # laplaciano tridiagonal L (diagonais inferior, principal e superior) e termo constante b
    inf_, dia, sup = np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)
    b = np.zeros(N)
    if esquerda[0] == "neumann":
        sup[0] = 2.0
        b[0] = -2 * esquerda[1] * dx
    if direita[0] == "neumann":
        inf_[-1] = 2.0
        b[-1] = 2 * direita[1] * dx
    inf_, dia, sup, b = inf_ / dx**2, dia / dx**2, sup / dx**2, b / dx**2
    s = alpha * dt
    # matriz do lado esquerdo, I - θ s L, em formato de banda
    ab = np.zeros((3, N))
    ab[0, 1:] = -theta * s * sup
    ab[1] = 1 - theta * s * dia
    ab[2, :-1] = -theta * s * inf_
    dirichlet = [(i, lado[1]) for i, lado in ((0, esquerda), (N - 1, direita)) if lado[0] == "dirichlet"]
    for i, _ in dirichlet:
        ab[1, i] = 1.0
        if i == 0:
            ab[0, 1] = 0.0
        else:
            ab[2, N - 2] = 0.0

    def Lu(v):
        w = dia * v + b
        w[:-1] += sup * v[1:]
        w[1:] += inf_ * v[:-1]
        return w

    n = int(np.ceil(t_final / dt - 1e-9))
    dt = t_final / n
    guardar = _instantes_guardados(instantes, t_final, dt, n)
    ts, us = [0.0], [u.copy()]
    for k in range(1, n + 1):
        t0 = (k - 1) * dt
        rhs = u + (1 - theta) * s * (Lu(u) - b) + s * b
        if fonte is not None:
            rhs += dt * ((1 - theta) * fonte(x, t0) + theta * fonte(x, t0 + dt))
        for i, valor in dirichlet:
            rhs[i] = valor
        u = solve_banded((1, 1), ab, rhs) if theta > 0 else rhs
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e100:
            return ResultadoEDP(x, np.array(ts + [k * dt]), np.array(us + [u.copy()]), k, False,
                                "a solução explodiu (instabilidade)", [dict(r=s / dx**2, dt=dt)])
        if k in guardar:
            ts.append(k * dt)
            us.append(u.copy())
    return ResultadoEDP(x, np.array(ts), np.array(us), n, True, "", [dict(r=s / dx**2, dt=dt)])


# ---- Poisson 2D -------------------------------------------------------------

def laplaciano_1d(n: int, dx: float, contorno: str = "dirichlet") -> sp.csr_matrix:
    """Segunda diferença 1D esparsa, n × n.

    "dirichlet": n nós interiores, com valor zero nas pontas (as pontas entram no lado direito).
    "neumann": n células (volumes finitos) com fluxo zero nas faces externas.
    """
    dia = -2 * np.ones(n)
    if contorno == "neumann":
        dia[[0, -1]] = -1
    elif contorno != "dirichlet":
        raise ValueError(f"contorno desconhecido: {contorno}")
    return sp.diags([np.ones(n - 1), dia, np.ones(n - 1)], [-1, 0, 1], format="csr") / dx**2


def laplaciano_2d(nx: int, ny: int, dx: float, dy: float, contorno: str = "dirichlet") -> sp.csr_matrix:
    """Laplaciano 2D esparso (estêncil de 5 pontos) como soma de Kronecker, ordem i*ny + j."""
    Lx, Ly = laplaciano_1d(nx, dx, contorno), laplaciano_1d(ny, dy, contorno)
    return (sp.kron(Lx, sp.identity(ny)) + sp.kron(sp.identity(nx), Ly)).tocsr()


def poisson_2d(f, g, Lx: float, Ly: float, nx: int, ny: int, metodo: str = "direto",
               omega: float | None = None, tol: float = 1e-8, max_iter: int = 100000) -> ResultadoEDP:
    """Resolve -∇²u = f no retângulo [0, Lx] × [0, Ly], com u = g no contorno.

    ``f`` e ``g`` são funções de (X, Y) (ou números). Malha de (nx+1) × (ny+1) nós.
    ``metodo``: "direto" (esparso), "jacobi", "gauss-seidel" ou "sor" (ordenação vermelho-preto);
    sem ``omega``, o SOR usa o ótimo do quadrado, 2/(1 + sin(π/n)).
    Os iterativos param quando a maior variação de u numa iteração fica abaixo de ``tol``;
    ``historico`` guarda essa variação a cada iteração.
    """
    x, y = np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = f(X, Y) if callable(f) else np.full_like(X, float(f))
    U = np.zeros_like(X)
    G = g(X, Y) if callable(g) else np.full_like(X, float(g))
    U[0, :], U[-1, :], U[:, 0], U[:, -1] = G[0, :], G[-1, :], G[:, 0], G[:, -1]
    if metodo == "direto":
        A = -laplaciano_2d(nx - 1, ny - 1, dx, dy)
        rhs = F[1:-1, 1:-1].copy()
        rhs[0, :] += U[0, 1:-1] / dx**2
        rhs[-1, :] += U[-1, 1:-1] / dx**2
        rhs[:, 0] += U[1:-1, 0] / dy**2
        rhs[:, -1] += U[1:-1, -1] / dy**2
        U[1:-1, 1:-1] = spsolve(A.tocsc(), rhs.ravel()).reshape(nx - 1, ny - 1)
        return ResultadoEDP((x, y), np.array([]), U, 1)
    cx, cy = dy**2 / (2 * (dx**2 + dy**2)), dx**2 / (2 * (dx**2 + dy**2))
    cf = dx**2 * dy**2 / (2 * (dx**2 + dy**2))
    if metodo == "sor" and omega is None:
        omega = 2 / (1 + np.sin(np.pi / max(nx, ny)))
    w = 1.0 if metodo == "gauss-seidel" else omega
    I, J = np.meshgrid(np.arange(1, nx), np.arange(1, ny), indexing="ij")
    cores = [(I + J) % 2 == 0, (I + J) % 2 == 1]
    historico = []

    def media(V):
        return cx * (V[2:, 1:-1] + V[:-2, 1:-1]) + cy * (V[1:-1, 2:] + V[1:-1, :-2]) + cf * F[1:-1, 1:-1]

    for it in range(1, max_iter + 1):
        antigo = U[1:-1, 1:-1].copy()
        if metodo == "jacobi":
            U[1:-1, 1:-1] = media(U)
        elif metodo in ("gauss-seidel", "sor"):
            for cor in cores:                   # vermelho-preto: cada cor só usa vizinhos da outra
                interior = U[1:-1, 1:-1]
                interior[cor] = (1 - w) * interior[cor] + w * media(U)[cor]
        else:
            raise ValueError(f"método desconhecido: {metodo}")
        variacao = float(np.max(np.abs(U[1:-1, 1:-1] - antigo)))
        historico.append(variacao)
        if variacao < tol:
            return ResultadoEDP((x, y), np.array([]), U, it, True, "", historico)
    return ResultadoEDP((x, y), np.array([]), U, max_iter, False, "máximo de iterações atingido", historico)


# ---- Burgers 1D -------------------------------------------------------------

def burgers(u0, x, t_final: float, dt: float, forma: str = "conservativa", periodico: bool = False,
            instantes=None) -> ResultadoEDP:
    """Integra u_t + (u²/2)_x = 0 (Burgers sem viscosidade) com esquemas de primeira ordem.

    "conservativa": volumes finitos com o fluxo de Godunov; os choques andam na velocidade certa.
    "nao-conservativa": upwind aplicado a u_t + u u_x = 0; os choques andam na velocidade errada.
    Estável para max|u| dt/dx <= 1. Com ``periodico=False`` as pontas ficam fixas.
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    u = np.array(u0, dtype=float)
    n = int(np.ceil(t_final / dt - 1e-9))
    dt = t_final / n
    lam = dt / dx
    guardar = _instantes_guardados(instantes, t_final, dt, n)
    ts, us = [0.0], [u.copy()]
    pontas = u[[0, -1]].copy()

    def godunov(ue, ud):
        # fluxo de Godunov para f(u) = u²/2 entre os estados à esquerda (ue) e à direita (ud)
        fe, fd = ue**2 / 2, ud**2 / 2
        choque = np.where((ue + ud) / 2 > 0, fe, fd)          # ue > ud: choque com velocidade (ue+ud)/2
        rarefacao = np.where(ue > 0, fe, np.where(ud < 0, fd, 0.0))
        return np.where(ue > ud, choque, rarefacao)

    for k in range(1, n + 1):
        um, up = np.roll(u, 1), np.roll(u, -1)
        if forma == "conservativa":
            u = u - lam * (godunov(u, up) - godunov(um, u))
        elif forma == "nao-conservativa":
            u = u - lam * np.where(u > 0, u * (u - um), u * (up - u))
        else:
            raise ValueError(f"forma desconhecida: {forma}")
        if not periodico:
            u[[0, -1]] = pontas
        if k in guardar:
            ts.append(k * dt)
            us.append(u.copy())
    return ResultadoEDP(x, np.array(ts), np.array(us), n, True, "", [dict(C=float(np.max(np.abs(u0))) * lam)])


# ---- elementos finitos 1D ---------------------------------------------------

def fem_1d(k, f, nos, ua: float, direita=("dirichlet", 0.0)) -> ResultadoEDP:
    """Elementos finitos lineares (Galerkin) para -(k u')' = f em [nos[0], nos[-1]].

    ``k``: função de x, avaliada no ponto médio de cada elemento (constante por elemento),
    ou número. ``f``: função de x (ou número); as cargas ∫ f φ_i dx usam Gauss-Legendre
    de 3 pontos em cada elemento.
    u(nos[0]) = ua; à direita, ("dirichlet", ub) ou ("neumann", q) com k u'(b) = q.
    ``historico`` guarda a matriz de rigidez global (esparsa) e o vetor de cargas.
    """
    nos = np.asarray(nos, dtype=float)
    N = nos.size
    h = np.diff(nos)
    meio = (nos[:-1] + nos[1:]) / 2
    ke = k(meio) * np.ones_like(h) if callable(k) else np.full_like(h, float(k))
    # rigidez de cada elemento: (k/h) [[1, -1], [-1, 1]]
    linhas = np.r_[np.arange(N - 1), np.arange(1, N), np.arange(N - 1), np.arange(1, N)]
    colunas = np.r_[np.arange(N - 1), np.arange(1, N), np.arange(1, N), np.arange(N - 1)]
    valores = np.r_[ke / h, ke / h, -ke / h, -ke / h]
    K = sp.coo_matrix((valores, (linhas, colunas)), shape=(N, N)).tocsr()
    # cargas: ∫ f φ dx em cada elemento, com φ_esq = 1 - ξ e φ_dir = ξ, ξ ∈ [0, 1]
    xi, pesos = np.polynomial.legendre.leggauss(3)
    xi, pesos = (xi + 1) / 2, pesos / 2
    F = np.zeros(N)
    for q, wq in zip(xi, pesos):
        xq = nos[:-1] + q * h
        fq = f(xq) * np.ones_like(xq) if callable(f) else np.full_like(xq, float(f))
        np.add.at(F, np.arange(N - 1), wq * h * fq * (1 - q))
        np.add.at(F, np.arange(1, N), wq * h * fq * q)
    Kc, Fc = K.tolil(), F.copy()
    if direita[0] == "neumann":
        Fc[-1] += direita[1]
        fixos = [(0, ua)]
    else:
        fixos = [(0, ua), (N - 1, direita[1])]
    for i, v in fixos:                     # impõe Dirichlet: troca a linha pela equação u_i = v
        Fc -= Kc[:, i].toarray().ravel() * v
        Kc[i, :] = 0
        Kc[:, i] = 0
        Kc[i, i] = 1.0
        Fc[i] = v
    u = spsolve(Kc.tocsc(), Fc)
    return ResultadoEDP(nos, np.array([]), u, 1, True, "", [K, F])
