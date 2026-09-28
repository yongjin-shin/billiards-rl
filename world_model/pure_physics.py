"""
world_model/pure_physics.py

Pooltool-free reimplementation of billiards ball motion and collision detection.
Pure numpy — no pooltool import at runtime.

Equations ported from pooltool source (MIT license).

The free-motion evolution chain (evolve_ball_motion, t_stop, and their helpers)
is numba @jit(nopython=True)'d. Plain Python+numpy was measured 2.4x SLOWER
than pooltool's evolve_ball_motion per call (pooltool's is itself numba-jitted,
so it runs near machine-code speed — a tiny (3,3)-array numpy call can't beat
that on interpreter+dispatch overhead alone). JIT-compiling our own formulas
recovers that without depending on pooltool's object model. The collision-time
functions below (np.roots-based) are NOT jitted — numba nopython mode doesn't
support np.roots, and event-time solving is out of scope for this replacement
(see roadmap.md).
"""

from __future__ import annotations

import numpy as np
from numba import jit
from numpy.typing import NDArray

# ── State constants (mirror pooltool.constants) ───────────────────────────────
STATIONARY = 0
SPINNING   = 1
SLIDING    = 2
ROLLING    = 3
POCKETED   = 4
NONTRANSLATING = frozenset({STATIONARY, SPINNING, POCKETED})

EPS = 1e-9


# ── Math helpers ──────────────────────────────────────────────────────────────
# jit(nopython=True): these + everything below up to t_stop() form the
# free-motion evolution call chain — no np.roots, all numba-compatible.

@jit(nopython=True, cache=True)
def _norm3d(v: NDArray) -> float:
    return float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))


@jit(nopython=True, cache=True)
def _unit_vector(v: NDArray) -> NDArray:
    n = _norm3d(v)
    return v / n if n > EPS else np.zeros(3, dtype=np.float64)


@jit(nopython=True, cache=True)
def _angle(v2: NDArray) -> float:
    ang = np.arctan2(v2[1], v2[0])
    return float(ang + 2 * np.pi if ang < 0 else ang)


@jit(nopython=True, cache=True)
def _rot(v: NDArray, phi: float) -> NDArray:
    """Rotate 3-column matrix or 3-vector by phi around z-axis."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ v


@jit(nopython=True, cache=True)
def _surface_velocity(rvw: NDArray, d: NDArray, R: float) -> NDArray:
    _, v, w = rvw
    return v + np.cross(w, R * d)


@jit(nopython=True, cache=True)
def _rel_velocity(rvw: NDArray, R: float) -> NDArray:
    return _surface_velocity(rvw, np.array([0.0, 0.0, -1.0]), R)


# ── Transition time calculations ─────────────────────────────────────────────

@jit(nopython=True, cache=True)
def get_slide_time(rvw: NDArray, R: float, u_s: float, g: float) -> float:
    if u_s == 0.0:
        return np.inf
    return 2 * _norm3d(_rel_velocity(rvw, R)) / (7 * u_s * g)


@jit(nopython=True, cache=True)
def get_roll_time(rvw: NDArray, u_r: float, g: float) -> float:
    if u_r == 0.0:
        return np.inf
    return _norm3d(rvw[1]) / (u_r * g)


@jit(nopython=True, cache=True)
def get_spin_time(rvw: NDArray, R: float, u_sp: float, g: float) -> float:
    if u_sp == 0.0:
        return np.inf
    return abs(rvw[2, 2]) * 2 / 5 * R / u_sp / g


# ── State evolution ───────────────────────────────────────────────────────────

@jit(nopython=True, cache=True)
def _evolve_spin_component(wz: float, R: float, u_sp: float, g: float, t: float) -> float:
    if t == 0 or abs(wz) < EPS:
        return wz
    alpha = 5 * u_sp * g / (2 * R)
    t_clamp = min(t, abs(wz) / alpha)
    sign = 1.0 if wz > 0 else -1.0
    return wz - sign * alpha * t_clamp


@jit(nopython=True, cache=True)
def _evolve_spin_state(rvw: NDArray, R: float, u_sp: float, g: float, t: float) -> NDArray:
    out = rvw.copy()
    out[2, 2] = _evolve_spin_component(rvw[2, 2], R, u_sp, g, t)
    return out


@jit(nopython=True, cache=True)
def _evolve_slide(rvw: NDArray, R: float, m: float, u_s: float, u_sp: float, g: float, t: float) -> NDArray:
    if t == 0:
        return rvw.copy()

    phi   = _angle(rvw[1])
    rvw_B = _rot(rvw.T, -phi).T                          # ball frame

    u_0 = _rot(_unit_vector(_rel_velocity(rvw, R)), -phi)

    out_B = np.zeros((3, 3), dtype=np.float64)
    out_B[0, 0] = rvw_B[1, 0] * t - 0.5 * u_s * g * t ** 2 * u_0[0]
    out_B[0, 1] = -0.5 * u_s * g * t ** 2 * u_0[1]
    out_B[0, 2] = 0.0
    out_B[1]    = rvw_B[1] - u_s * g * t * u_0
    out_B[2]    = rvw_B[2] - (5 / (2 * R)) * u_s * g * t * np.cross(u_0, np.array([0, 0, 1.0]))
    out_B[2, 2] = rvw_B[2, 2]
    out_B       = _evolve_spin_state(out_B, R, u_sp, g, t)

    out_T       = _rot(out_B.T, phi).T
    out_T[0]   += rvw[0]
    return out_T


@jit(nopython=True, cache=True)
def _evolve_roll(rvw: NDArray, R: float, u_r: float, u_sp: float, g: float, t: float) -> NDArray:
    if t == 0:
        return rvw.copy()

    r_0, v_0, _ = rvw
    v_hat = _unit_vector(v_0)

    r = r_0 + v_0 * t - 0.5 * u_r * g * t ** 2 * v_hat
    v = v_0 - u_r * g * t * v_hat
    w = _rot(v, np.pi / 2) / R

    # independent z-spin decay
    temp   = _evolve_spin_state(rvw, R, u_sp, g, t)
    w[2]   = temp[2, 2]

    out    = np.empty((3, 3), dtype=np.float64)
    out[0] = r
    out[1] = v
    out[2] = w
    return out


@jit(nopython=True, cache=True)
def evolve_ball_motion(
    state : int,
    rvw   : NDArray,
    R: float, m: float, u_s: float, u_sp: float, u_r: float, g: float,
    t     : float,
) -> tuple[NDArray, int]:
    """Evolve ball by time t. Returns (new_rvw, new_state). No pooltool dependency."""
    rvw = np.asarray(rvw).astype(np.float64)

    if state == STATIONARY or state == POCKETED:
        return rvw, state

    if state == SLIDING:
        dt = get_slide_time(rvw, R, u_s, g)
        if t >= dt:
            rvw   = _evolve_slide(rvw, R, m, u_s, u_sp, g, dt)
            state = ROLLING
            t    -= dt
        else:
            return _evolve_slide(rvw, R, m, u_s, u_sp, g, t), SLIDING

    if state == ROLLING:
        dt = get_roll_time(rvw, u_r, g)
        if t >= dt:
            rvw   = _evolve_roll(rvw, R, u_r, u_sp, g, dt)
            state = SPINNING
            t    -= dt
        else:
            return _evolve_roll(rvw, R, u_r, u_sp, g, t), ROLLING

    if state == SPINNING:
        dt = get_spin_time(rvw, R, u_sp, g)
        if t >= dt:
            return _evolve_spin_state(rvw, R, u_sp, g, dt), STATIONARY
        else:
            return _evolve_spin_state(rvw, R, u_sp, g, t), SPINNING

    raise ValueError("Unknown state")


@jit(nopython=True, cache=True)
def t_stop(rvw: NDArray, state: int, R: float, u_s: float, u_sp: float, u_r: float, g: float) -> float:
    """Total time until ball comes to rest."""
    total = 0.0
    rvw = np.asarray(rvw).astype(np.float64)

    if state == SLIDING:
        dt = get_slide_time(rvw, R, u_s, g)
        rvw = _evolve_slide(rvw, R, 1.0, u_s, u_sp, g, dt)
        state = ROLLING
        total += dt

    if state == ROLLING:
        dt = get_roll_time(rvw, u_r, g)
        rvw = _evolve_roll(rvw, R, u_r, u_sp, g, dt)
        state = SPINNING
        total += dt

    if state == SPINNING:
        total += get_spin_time(rvw, R, u_sp, g)

    return total


# ── Collision time helpers ────────────────────────────────────────────────────

def _get_u(rvw: NDArray, R: float, phi: float, s: int) -> NDArray:
    if s == ROLLING:
        return np.array([1.0, 0.0, 0.0])
    rv = _rel_velocity(rvw, R)
    if np.all(np.abs(rv) < EPS):
        return np.array([1.0, 0.0, 0.0])
    return _rot(_unit_vector(rv), -phi)


def _min_positive_real_root(coeffs: tuple[float, ...], eps: float = 1e-10) -> float:
    """Smallest positive real root of polynomial with coefficients [a_n, ..., a_0]."""
    c = [x for x in coeffs]
    while len(c) > 1 and abs(c[0]) < eps:
        c.pop(0)
    if len(c) <= 1:
        return np.inf

    roots = np.roots(c)
    valid = roots[(np.abs(roots.imag) <= eps) & (roots.real > eps)].real
    return float(valid.min()) if len(valid) else np.inf


# ── Collision coefficient calculators ─────────────────────────────────────────

def _sliding_rolling_coeffs(rvw: NDArray, R: float, mu: float, g: float):
    """Returns (ax, ay, bx, by) for a sliding or rolling ball trajectory."""
    phi = _angle(rvw[1])
    v   = _norm3d(rvw[1])
    u   = _get_u(rvw, R, phi, rvw if False else SLIDING)   # placeholder; caller passes s
    return phi, v, mu, g


def _traj_coeffs(rvw: NDArray, s: int, R: float, mu: float, g: float):
    """Trajectory acceleration and velocity coefficients in table frame."""
    if s in NONTRANSLATING:
        return 0.0, 0.0, 0.0, 0.0

    phi = _angle(rvw[1])
    v   = _norm3d(rvw[1])
    u   = _get_u(rvw, R, phi, s)
    K   = -0.5 * mu * g
    cp, sp = np.cos(phi), np.sin(phi)

    ax = K * (u[0] * cp - u[1] * sp)
    ay = K * (u[0] * sp + u[1] * cp)
    bx = v * cp
    by = v * sp
    return ax, ay, bx, by


# ── Public collision time functions ───────────────────────────────────────────

def ball_ball_collision_time(
    rvw1: NDArray, rvw2: NDArray,
    s1: int, s2: int,
    mu1: float, mu2: float,
    m1: float, m2: float,
    g1: float, g2: float,
    R: float,
) -> float:
    c1x, c1y = rvw1[0, 0], rvw1[0, 1]
    c2x, c2y = rvw2[0, 0], rvw2[0, 1]

    # Skip if balls are already intersecting
    dist = np.sqrt((c2x - c1x) ** 2 + (c2y - c1y) ** 2)
    if dist < 2 * R - EPS:
        return np.inf

    a1x, a1y, b1x, b1y = _traj_coeffs(rvw1, s1, R, mu1, g1)
    a2x, a2y, b2x, b2y = _traj_coeffs(rvw2, s2, R, mu2, g2)

    Ax, Ay = a2x - a1x, a2y - a1y
    Bx, By = b2x - b1x, b2y - b1y
    Cx, Cy = c2x - c1x, c2y - c1y

    a = Ax ** 2 + Ay ** 2
    b = 2 * Ax * Bx + 2 * Ay * By
    c = Bx ** 2 + 2 * Ax * Cx + 2 * Ay * Cy + By ** 2
    d = 2 * Bx * Cx + 2 * By * Cy
    e = Cx ** 2 + Cy ** 2 - 4 * R ** 2

    coeffs = (a, b, c, d, e)
    c_list = list(coeffs)
    while len(c_list) > 1 and abs(c_list[0]) < EPS:
        c_list.pop(0)
    if len(c_list) <= 1:
        return np.inf

    roots = np.roots(c_list)
    valid = roots[(np.abs(roots.imag) <= 1e-10) & (roots.real > EPS)].real

    # Filter spurious roots: polynomial extends the parabolic trajectory past state
    # transitions (e.g. ball decelerates, stops, then "reappears" on the other side).
    # Physical validation via evolve_ball_motion correctly handles state transitions.
    min_t = np.inf
    for t in sorted(valid):
        rvw1_t, _ = evolve_ball_motion(s1, rvw1, R, m1, mu1, 1.0, mu1, g1, t)
        rvw2_t, _ = evolve_ball_motion(s2, rvw2, R, m2, mu2, 1.0, mu2, g2, t)
        actual_dist = float(np.linalg.norm(rvw1_t[0, :2] - rvw2_t[0, :2]))
        if abs(actual_dist - 2 * R) < 1e-4:
            min_t = t
            break

    return min_t


def ball_linear_cushion_time(
    rvw: NDArray, s: int,
    lx: float, ly: float, l0: float,
    p1: NDArray, p2: NDArray,
    direction: int,
    mu: float, m: float, g: float, R: float,
) -> float:
    if s in NONTRANSLATING:
        return np.inf

    ax, ay, bx, by = _traj_coeffs(rvw, s, R, mu, g)
    cx, cy = rvw[0, 0], rvw[0, 1]

    A = lx * ax + ly * ay
    B = lx * bx + ly * by
    seg_norm = np.sqrt(lx ** 2 + ly ** 2)

    def _quad_roots(C: float) -> list[float]:
        if abs(A) < EPS:
            if abs(B) < EPS:
                return []
            t = -C / B
            return [t]
        bp = B / 2
        disc = bp ** 2 - A * C
        if disc < 0:
            return []
        sq = np.sqrt(max(disc, 0.0))
        return [(-bp - sq) / A, (-bp + sq) / A]

    if direction == 0:
        all_roots = _quad_roots(l0 + lx * cx + ly * cy + R * seg_norm)
    elif direction == 1:
        all_roots = _quad_roots(l0 + lx * cx + ly * cy - R * seg_norm)
    else:
        all_roots  = _quad_roots(l0 + lx * cx + ly * cy + R * seg_norm)
        all_roots += _quad_roots(l0 + lx * cx + ly * cy - R * seg_norm)

    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    seg = p2 - p1
    seg_len_sq = float(np.dot(seg, seg))

    ball_t_stop = t_stop(rvw, s, R, mu, 1.0, mu, g)

    min_t = np.inf
    for root in all_roots:
        if not np.isfinite(root) or root <= EPS:
            continue
        # Reject roots beyond when the ball actually stops — polynomial extrapolates
        # the trajectory past state transitions, producing spurious roots.
        if root > ball_t_stop + 1e-6:
            continue
        # Boundary check: is the collision point within segment p1→p2?
        rvw_t, _ = evolve_ball_motion(s, rvw, R, m, mu, 1.0, mu, g, root)
        s_score = -float(np.dot(p1 - rvw_t[0, :2], seg)) / seg_len_sq
        if 0.0 <= s_score <= 1.0:
            min_t = min(min_t, root)

    return min_t


def ball_circular_cushion_time(
    rvw: NDArray, s: int,
    a: float, b: float, r: float,
    mu: float, m: float, g: float, R: float,
) -> float:
    if s in NONTRANSLATING:
        return np.inf

    ax, ay, bx, by = _traj_coeffs(rvw, s, R, mu, g)
    cx, cy = rvw[0, 0], rvw[0, 1]

    A = 0.5 * (ax ** 2 + ay ** 2)
    B = ax * bx + ay * by
    C = ax * (cx - a) + ay * (cy - b) + 0.5 * (bx ** 2 + by ** 2)
    D = bx * (cx - a) + by * (cy - b)
    E = 0.5 * (a ** 2 + b ** 2 + cx ** 2 + cy ** 2 - (r + R) ** 2) - (cx * a + cy * b)

    return _min_positive_real_root((A, B, C, D, E))


def ball_pocket_time(
    rvw: NDArray, s: int,
    a: float, b: float, r: float,
    mu: float, m: float, g: float, R: float,
) -> float:
    if s in NONTRANSLATING:
        return np.inf

    ax, ay, bx, by = _traj_coeffs(rvw, s, R, mu, g)
    cx, cy = rvw[0, 0], rvw[0, 1]

    A = 0.5 * (ax ** 2 + ay ** 2)
    B = ax * bx + ay * by
    C = ax * (cx - a) + ay * (cy - b) + 0.5 * (bx ** 2 + by ** 2)
    D = bx * (cx - a) + by * (cy - b)
    E = 0.5 * (a ** 2 + b ** 2 + cx ** 2 + cy ** 2 - r ** 2) - (cx * a + cy * b)

    return _min_positive_real_root((A, B, C, D, E))
