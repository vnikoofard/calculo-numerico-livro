"""Equações diferenciais ordinárias (caps. 12, 13 e 14).

Implementações de REFERÊNCIA, curtas e legíveis:

- passo fixo explícito: Euler, Heun (RK2), RK4 (:func:`passo_fixo`);
- passo adaptativo com o par de Dormand-Prince 5(4) (:func:`rk45`), o mesmo de ``solve_ivp``;
- implícitos: Euler implícito e trapézio, com Newton (:func:`implicito`);
- passo múltiplo: Adams-Bashforth de 2 passos (:func:`adams_bashforth2`);
- simplético: Verlet de velocidade (:func:`verlet`);
- problemas de contorno: tiro (:func:`tiro`), diferenças finitas com Newton
  (:func:`contorno_dif_finitas`) e autoestados de Schrödinger 1D (:func:`autoestados`).

Os integradores devolvem :class:`ResultadoEDO` com a malha de tempo, a solução e
o número de avaliações de f. Para uso em produção: ``scipy.integrate.solve_ivp``
e ``scipy.integrate.solve_bvp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ResultadoEDO:
    t: np.ndarray                     # instantes t_0, t_1, ..., t_N
    y: np.ndarray                     # y[i] é a solução em t[i] (forma (N+1, m))
    avaliacoes: int                   # número de avaliações de f
    sucesso: bool = True
    motivo: str = ""
    rejeitados: int = 0               # passos rejeitados (métodos adaptativos)
    historico: list = field(default_factory=list)  # informação extra (iterações de Newton, tiros...)

    def __repr__(self) -> str:
        estado = "sucesso" if self.sucesso else f"FALHOU ({self.motivo})"
        return (f"ResultadoEDO({estado}, passos={len(self.t) - 1}, avaliações={self.avaliacoes}, "
                f"rejeitados={self.rejeitados})")


def _vetor(y0) -> np.ndarray:
    return np.atleast_1d(np.asarray(y0, dtype=float))


# ---- passo fixo explícito ---------------------------------------------------

def _euler(f, t, y, h):
    return y + h * f(t, y), 1


def _heun(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + h / 2 * (k1 + k2), 2


def _rk4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 4


METODOS_FIXOS = {"euler": _euler, "heun": _heun, "rk4": _rk4}


def passo_fixo(f: Callable, t_span, y0, h: float, metodo: str = "rk4") -> ResultadoEDO:
    """Integra y' = f(t, y) de t_span[0] a t_span[1] com passo fixo h.

    ``metodo`` é "euler" (ordem 1), "heun" (ordem 2) ou "rk4" (ordem 4).
    O último passo é encurtado para terminar exatamente em t_span[1].
    """
    passo = METODOS_FIXOS[metodo]
    t0, tf = map(float, t_span)
    n = int(np.ceil((tf - t0) / h - 1e-9))
    ts = [t0]
    ys = [_vetor(y0)]
    nav = 0
    for _ in range(n):
        hi = min(h, tf - ts[-1])
        y_novo, k = passo(f, ts[-1], ys[-1], hi)
        nav += k
        ts.append(ts[-1] + hi)
        ys.append(y_novo)
        if not np.all(np.isfinite(y_novo)):
            return ResultadoEDO(np.array(ts), np.array(ys), nav, False, "a solução explodiu (inf ou NaN)")
    return ResultadoEDO(np.array(ts), np.array(ys), nav)


# ---- passo adaptativo: Dormand-Prince 5(4) ------------------------------------

_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def rk45(f: Callable, t_span, y0, rtol: float = 1e-6, atol: float = 1e-9, h0: float | None = None,
         h_max: float = np.inf, max_passos: int = 100_000) -> ResultadoEDO:
    """Runge-Kutta adaptativo de Dormand-Prince 5(4), com controle do erro local.

    Em cada passo, a diferença entre as soluções de ordem 5 e 4 estima o erro.
    O passo é aceito se ``|erro| <= atol + rtol*|y|`` (norma RMS) e o novo passo é
    ``h * 0,9 * (1/erro)^(1/5)``, limitado entre 0,2h e 5h. A solução que avança
    é a de ordem 5 (extrapolação local), como em ``solve_ivp(method="RK45")``.
    """
    t0, tf = map(float, t_span)
    y = _vetor(y0)
    k1 = f(t0, y)
    nav = 1
    if h0 is None:
        escala = atol + rtol * np.abs(y)
        d0, d1 = np.sqrt(np.mean((y / escala) ** 2)), np.sqrt(np.mean((k1 / escala) ** 2))
        h0 = 1e-6 if min(d0, d1) < 1e-5 else 0.01 * d0 / d1
    h = min(h0, h_max, tf - t0)
    ts, ys = [t0], [y.copy()]
    rejeitados = 0
    t = t0
    for _ in range(max_passos):
        if t >= tf:
            break
        h = min(h, tf - t)
        k = [k1]
        for i in range(1, 7):
            yi = y + h * sum(a * kj for a, kj in zip(_A[i], k))
            k.append(f(t + _C[i] * h, yi))
        nav += 6
        y5 = y + h * sum(b * kj for b, kj in zip(_B5, k))
        erro_vet = h * sum((b5 - b4) * kj for b5, b4, kj in zip(_B5, _B4, k))
        escala = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        erro = np.sqrt(np.mean((erro_vet / escala) ** 2))
        if erro <= 1.0:
            t += h
            y = y5
            k1 = k[6]                 # FSAL: o último estágio é o primeiro do próximo passo
            ts.append(t)
            ys.append(y.copy())
            fator = 5.0 if erro == 0 else min(5.0, 0.9 * erro ** -0.2)
        else:
            rejeitados += 1
            fator = max(0.2, 0.9 * erro ** -0.2)
        h = min(h * fator, h_max)
        if h < 1e-14 * max(1.0, abs(t)):
            return ResultadoEDO(np.array(ts), np.array(ys), nav, False, "passo pequeno demais", rejeitados)
    else:
        return ResultadoEDO(np.array(ts), np.array(ys), nav, False, "número máximo de passos", rejeitados)
    return ResultadoEDO(np.array(ts), np.array(ys), nav, True, "", rejeitados)


# ---- implícitos ---------------------------------------------------------------

def _jacobiana_numerica(f, t, y, fy):
    m = y.size
    J = np.empty((m, m))
    for j in range(m):
        d = 1e-7 * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += d
        J[:, j] = (f(t, yp) - fy) / d
    return J


def implicito(f: Callable, t_span, y0, h: float, metodo: str = "euler", jac: Callable | None = None,
              tol: float = 1e-10, max_newton: int = 20) -> ResultadoEDO:
    """Euler implícito (ordem 1) ou trapézio implícito (ordem 2), com passo fixo h.

    Cada passo resolve uma equação não linear em y_{n+1} pelo método de Newton
    (cap. 5), com a jacobiana ``jac(t, y)`` ou uma aproximação por diferenças.
    ``historico`` guarda o número de iterações de Newton de cada passo.
    """
    if metodo not in ("euler", "trapezio"):
        raise ValueError('metodo deve ser "euler" ou "trapezio".')
    teta = 1.0 if metodo == "euler" else 0.5
    t0, tf = map(float, t_span)
    n = int(np.ceil((tf - t0) / h - 1e-9))
    ts, ys = [t0], [_vetor(y0)]
    nav = 0
    iters = []
    m = ys[0].size
    for _ in range(n):
        t, y = ts[-1], ys[-1]
        hi = min(h, tf - t)
        fy = f(t, y)
        nav += 1
        z = y + hi * fy                      # chute: um passo de Euler explícito
        for it in range(1, max_newton + 1):
            fz = f(t + hi, z)
            nav += 1
            G = z - y - hi * (teta * fz + (1 - teta) * fy)
            if jac is not None:
                J = np.atleast_2d(jac(t + hi, z))
            else:
                J = _jacobiana_numerica(f, t + hi, z, fz)
                nav += m
            dz = np.linalg.solve(np.eye(m) - hi * teta * J, -G)
            z = z + dz
            if np.linalg.norm(dz) <= tol * (1 + np.linalg.norm(z)):
                break
        else:
            return ResultadoEDO(np.array(ts), np.array(ys), nav, False, "Newton não convergiu", 0, iters)
        iters.append(it)
        ts.append(t + hi)
        ys.append(z)
    return ResultadoEDO(np.array(ts), np.array(ys), nav, True, "", 0, iters)


# ---- passo múltiplo -------------------------------------------------------------

def adams_bashforth2(f: Callable, t_span, y0, h: float) -> ResultadoEDO:
    """Adams-Bashforth de 2 passos (ordem 2): y_{n+1} = y_n + h(3f_n - f_{n-1})/2.

    O primeiro passo é dado por Heun. Cada passo seguinte custa UMA avaliação de f.
    """
    t0, tf = map(float, t_span)
    n = int(round((tf - t0) / h))
    ts = t0 + h * np.arange(n + 1)
    y = np.empty((n + 1, _vetor(y0).size))
    y[0] = _vetor(y0)
    f_ant = f(ts[0], y[0])
    y[1], _ = _heun(f, ts[0], y[0], h)
    nav = 2
    for i in range(1, n):
        f_at = f(ts[i], y[i])
        nav += 1
        y[i + 1] = y[i] + h / 2 * (3 * f_at - f_ant)
        f_ant = f_at
    return ResultadoEDO(ts, y, nav)


# ---- simplético -----------------------------------------------------------------

def verlet(aceleracao: Callable, t_span, q0, v0, h: float) -> ResultadoEDO:
    """Verlet de velocidade para q'' = a(q): simplético, ordem 2, uma avaliação de a por passo.

    Devolve y com as colunas [q..., v...].
    """
    t0, tf = map(float, t_span)
    n = int(round((tf - t0) / h))
    q = _vetor(q0)
    v = _vetor(v0)
    a = aceleracao(q)
    ys = np.empty((n + 1, 2 * q.size))
    ys[0] = np.r_[q, v]
    for i in range(n):
        v_meio = v + h / 2 * a
        q = q + h * v_meio
        a = aceleracao(q)
        v = v_meio + h / 2 * a
        ys[i + 1] = np.r_[q, v]
    return ResultadoEDO(t0 + h * np.arange(n + 1), ys, n + 1)


# ---- problemas de contorno ------------------------------------------------------

def tiro(f: Callable, x_span, ya: float, residuo: Callable, s0: float, s1: float, tol: float = 1e-10,
         max_iter: int = 50, rtol_edo: float = 1e-10, atol_edo: float = 1e-12) -> ResultadoEDO:
    """Método do tiro para y'' = f(x, y, y') com y(a) = ya e uma condição em b.

    O sistema y' = v, v' = f é integrado com ``solve_ivp`` (RK45) a partir de
    y(a) = ya, y'(a) = s. ``residuo(y_b, v_b)`` deve se anular na solução: por
    exemplo ``lambda y, v: y - yb`` (Dirichlet) ou ``lambda y, v: v`` (ponta isolada).
    O ângulo de tiro s é ajustado pelo método da secante a partir de s0 e s1;
    ``historico`` guarda os pares (s, resíduo) de cada tiro.
    """
    from scipy.integrate import solve_ivp

    a, b = map(float, x_span)

    def sistema(x, u):
        return [u[1], f(x, u[0], u[1])]

    def disparar(s):
        sol = solve_ivp(sistema, (a, b), [ya, s], rtol=rtol_edo, atol=atol_edo, dense_output=True)
        return sol, residuo(sol.y[0, -1], sol.y[1, -1])

    nav = 0
    sol0, r0 = disparar(s0)
    sol1, r1 = disparar(s1)
    nav += sol0.nfev + sol1.nfev
    hist = [(s0, r0), (s1, r1)]
    sucesso, motivo = False, "número máximo de tiros"
    for _ in range(max_iter):
        if r1 == r0:
            motivo = "resíduo igual em dois tiros (secante sem inclinação)"
            break
        s0, s1 = s1, s1 - r1 * (s1 - s0) / (r1 - r0)
        r0 = r1
        sol1, r1 = disparar(s1)
        nav += sol1.nfev
        hist.append((s1, r1))
        if not np.isfinite(r1):
            motivo = "o tiro explodiu"
            break
        if abs(s1 - s0) <= tol * (1 + abs(s1)):
            # o passo parou; só é sucesso se o resíduo de fato zerou
            sucesso = abs(r1) <= 1e-6 * max(1.0, abs(hist[0][1]))
            motivo = "" if sucesso else "a secante parou, mas o resíduo não zerou"
            break
    x = np.linspace(a, b, 201)
    return ResultadoEDO(x, sol1.sol(x).T, nav, sucesso, motivo, 0, hist)


def contorno_dif_finitas(f: Callable, x_span, ya: float, direita: tuple, n: int, chute=None,
                         tol: float = 1e-10, max_iter: int = 30) -> ResultadoEDO:
    """Diferenças finitas centradas (ordem 2) para y'' = f(x, y, y') em n subintervalos.

    Condição à esquerda: y(a) = ya. À direita, ``direita`` é ("dirichlet", yb)
    ou ("neumann", y'b), esta última com um nó fantasma. O sistema não linear
    resultante é tridiagonal e é resolvido por Newton, com a jacobiana calculada
    por diferenças (f pode ser não linear). Para f linear, Newton termina em 1 iteração.
    ``t`` guarda os nós x_i e ``historico`` a norma da correção de Newton.
    """
    from scipy.linalg import solve_banded

    a, b = map(float, x_span)
    tipo, valor = direita
    if tipo not in ("dirichlet", "neumann"):
        raise ValueError('direita deve ser ("dirichlet", yb) ou ("neumann", dyb).')
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    y = np.linspace(ya, valor if tipo == "dirichlet" else ya, n + 1) if chute is None else np.array(chute, float)
    y[0] = ya
    if tipo == "dirichlet":
        y[-1] = valor
    incog = slice(1, n) if tipo == "dirichlet" else slice(1, n + 1)
    nav = 0
    hist = []

    def residuo(y):
        # estende com o nó fantasma y_{n+1} = y_{n-1} + 2h y'b (Neumann)
        ye = np.r_[y, y[-2] + 2 * h * valor] if tipo == "neumann" else y
        dy = (ye[2:] - ye[:-2]) / (2 * h)
        d2y = (ye[2:] - 2 * ye[1:-1] + ye[:-2]) / h**2
        return d2y - f(x[1:len(ye) - 1], ye[1:-1], dy)

    for it in range(max_iter):
        with np.errstate(all="ignore"):
            R = residuo(y)
        nav += 1
        if not np.all(np.isfinite(R)):
            return ResultadoEDO(x, y, nav, False, "Newton divergiu (valores não finitos)", 0, hist)
        m = R.size
        # jacobiana tridiagonal por diferenças: perturba nós de 3 em 3
        J = np.zeros((3, m))                      # formato de banda para solve_banded
        livres = np.arange(1, 1 + m)
        for desloc in range(3):
            idx = livres[desloc::3]
            if idx.size == 0:
                continue
            yp = y.copy()
            d = 1e-7 * np.maximum(1.0, np.abs(y[idx]))
            yp[idx] += d
            with np.errstate(all="ignore"):
                dR = (residuo(yp) - R)
            nav += 1
            for j, dj in zip(idx, d):
                col = j - 1
                for lin in (col - 1, col, col + 1):
                    if 0 <= lin < m:
                        J[1 + lin - col, col] = dR[lin] / dj
        if not np.all(np.isfinite(J)):
            return ResultadoEDO(x, y, nav, False, "Newton divergiu (valores não finitos)", 0, hist)
        dy = solve_banded((1, 1), J, -R)
        y[incog] += dy
        hist.append(float(np.max(np.abs(dy))))
        if hist[-1] <= tol * (1 + np.max(np.abs(y))):
            return ResultadoEDO(x, y, nav, True, "", 0, hist)
    return ResultadoEDO(x, y, nav, False, "Newton não convergiu", 0, hist)


def autoestados(V: Callable, x_span, n: int, k: int = 4, coef: float = 0.5):
    """Os k menores níveis de energia de -coef·ψ'' + V(x)ψ = Eψ com ψ = 0 nas pontas.

    Com coef = 1/2, são as unidades em que ħ = m = 1; para um elétron de massa
    efetiva m*, com x em nm e E em eV, coef = ħ²/(2m*) = 0,0381/(m*/mₑ) eV·nm².
    Diferenças centradas em n subintervalos dão uma matriz tridiagonal simétrica
    (cap. 8). Devolve (energias, x, psi), com as funções de onda normalizadas em psi[:, j].
    """
    from scipy.linalg import eigh_tridiagonal

    a, b = map(float, x_span)
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    xi = x[1:-1]
    diag = 2 * coef / h**2 + V(xi)
    fora = np.full(xi.size - 1, -coef / h**2)
    E, psi = eigh_tridiagonal(diag, fora, select="i", select_range=(0, k - 1))
    psi_completo = np.zeros((n + 1, k))
    psi_completo[1:-1] = psi / np.sqrt(h)
    return E, x, psi_completo
