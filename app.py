from __future__ import annotations
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# ========================= Shared Helpers =========================

def compute_trajectory(num_points: int, phase: float, axes=(65.0, 50.0, 35.0)):
    """Ellipsoidal path; returns x,y,z,s progress (0..1)."""
    a, b, c = axes
    s = np.linspace(0.0, 1.0, num_points)
    theta_s = np.pi * s
    phi_s   = 2*np.pi * s
    x = a * np.sin(theta_s + phase) * np.cos(phi_s)
    y = b * np.sin(theta_s + phase) * np.sin(phi_s)
    z = c * np.cos(theta_s + phase)
    return x, y, z, s

def power_intensity_ellipsoidal(x, y, z, axes=(65.0,50.0,35.0), P0=1.0, soft_core=0.08):
    a, b, c = axes
    r2e = (x/a)**2 + (y/b)**2 + (z/c)**2
    return P0 / (r2e + soft_core)  # normalized later

def fibonacci_directions(n: int):
    ga = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.clip(1.0 - z*z, 0.0, 1.0))
    phi = ga * k
    ux, uy, uz = r * np.cos(phi), r * np.sin(phi), z
    return ux, uy, uz

def ellipsoid_intersection_scale(ux, uy, uz, a, b, c):
    # (t ux / a)^2 + (t uy / b)^2 + (t uz / c)^2 = 1  -> t = 1/sqrt((ux/a)^2+...)
    denom = (ux/a)**2 + (uy/b)**2 + (uz/c)**2
    denom = np.where(denom <= 1e-15, 1e-15, denom)
    return 1.0 / np.sqrt(denom)

# ========================= CTC / Spinning String =========================
class CFG:
    Gmu = 2.0e-2   # exaggerated for visibility
    beta = 0.6     # twist parameter
    geodesic_type = "null"  # or "timelike"
    E = 1.0
    n_rays = 12
    L_span = (-12.0, 12.0)
    r0 = 30.0
    phi0 = np.pi - 0.05
    sgn_r = -1
    n_steps = 900
    dlam = 0.02


def integrate_geodesic(alpha: float, beta: float, E: float, L_eff: float,
                       r0: float, phi0: float, sgn_r: int, n_steps: int, dlam: float, kappa: float):
    t  = np.zeros(n_steps + 1)
    r  = np.zeros(n_steps + 1)
    ph = np.zeros(n_steps + 1)
    r[0] = r0; ph[0] = phi0; t[0] = 0.0
    sgn = -1.0 if sgn_r <= 0 else 1.0
    x = np.zeros_like(r); y = np.zeros_like(r)
    valid = np.ones_like(r, dtype=bool)
    for i in range(n_steps):
        ri = r[i]
        if ri <= 0:
            valid[i:] = False; break
        denom = max(alpha*alpha*ri*ri, 1e-15)
        phi_dot = L_eff / denom
        rad_inside = E*E + kappa - (L_eff*L_eff)/(alpha*alpha*ri*ri)
        if rad_inside < 0:
            sgn *= -1.0
            rad_inside = abs(rad_inside)
        r_dot = sgn * np.sqrt(rad_inside)
        t_dot = E - beta * phi_dot
        r[i+1]  = ri + r_dot * dlam
        ph[i+1] = ph[i] + phi_dot * dlam
        t[i+1]  = t[i] + t_dot * dlam
        if r[i+1] > r0 + 5.0:
            valid[i+1:] = False; break
    x[:] = r * np.cos(ph); y[:] = r * np.sin(ph)
    return t, r, ph, x, y, valid

# ========================= Time‑Time (helical on cone) =========================

def conical_phi(phi, delta):
    alpha = delta / (2.0*np.pi)
    return (1.0 - alpha) * phi

def helical_on_cone(num_points, r0, amp, turns, delta, phase=0.0, z_span=25.0):
    s = np.linspace(0.0, 1.0, num_points)
    r   = r0 + amp * np.sin(2*np.pi*s + phase)
    phi = 2*np.pi*turns*s + phase
    phi_cone = conical_phi(phi, delta)
    theta = 0.5*np.pi + 0.2*np.sin(4*np.pi*s + 0.3*phase)
    z_lin = -0.5*z_span + z_span*s
    x = r * np.sin(theta) * np.cos(phi_cone)
    y = r * np.sin(theta) * np.sin(phi_cone)
    z_sph = r * np.cos(theta)
    z = 0.4*z_sph + 0.6*z_lin
    return np.column_stack([x, y, z])

def unit_vectors(v):
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1.0
    return v / n

def direction_change_indices(points, angle_threshold_deg=15.0):
    p = np.asarray(points)
    v = np.diff(p, axis=0)
    u = unit_vectors(v)
    cosang = np.sum(u[1:] * u[:-1], axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    idx = np.where(ang > angle_threshold_deg)[0] + 1
    return idx.tolist()

# ========================= Routes =========================

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/warp")
def api_warp():
    a = float(request.args.get("a", 65.0))
    b = float(request.args.get("b", 50.0))
    c = float(request.args.get("c", 35.0))
    E = float(request.args.get("E", 1.5))
    rho = float(request.args.get("rho", 0.6))
    m = float(request.args.get("m", 1.0))
    classical = request.args.get("classical", "false").lower() == "true"
    n_traj = int(request.args.get("n_traj", 10))
    n_pts  = int(request.args.get("n_pts", 300))
    phases = np.linspace(0.0, 2*np.pi, n_traj, endpoint=False)

    # Ellipsoid wireframe lines
    rings = 28; stacks = 14
    u = np.linspace(0, 2*np.pi, rings)
    v = np.linspace(0, np.pi, stacks)
    ell_lines = []
    for vv in v:  # parallels
        x = a * np.cos(u) * np.sin(vv)
        y = b * np.sin(u) * np.sin(vv)
        z = c * np.cos(vv) * np.ones_like(u)
        ell_lines.append({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})
    for uu in u:  # meridians
        x = a * np.cos(uu) * np.sin(v)
        y = b * np.sin(uu) * np.sin(v)
        z = c * np.cos(v)
        ell_lines.append({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})

    # Power lines (radial) to ellipsoid hull
    plines = []
    ux, uy, uz = fibonacci_directions(20)
    t_end = ellipsoid_intersection_scale(ux, uy, uz, a, b, c)
    for i in range(len(ux)):
        x_end, y_end, z_end = t_end[i]*ux[i], t_end[i]*uy[i], t_end[i]*uz[i]
        plines.append({"x": [0.0, x_end], "y": [0.0, y_end], "z": [0.0, z_end]})

    # Trajectories + qubits + speed coloring
    trajs = []
    c_light = 1.0
    for ph in phases:
        x, y, z, s = compute_trajectory(n_pts, ph, axes=(a,b,c))
        r2e = (x/a)**2 + (y/b)**2 + (z/c)**2
        I = power_intensity_ellipsoidal(x, y, z, axes=(a,b,c), P0=1.0, soft_core=0.08)
        K = E - rho * (r2e)
        K = np.clip(K, 0.0, None)
        if classical:
            v = np.sqrt(2.0*np.maximum(K, 0.0)/max(m, 1e-12))
            vfrac = v / (c_light + 1e-12)
        else:
            gamma = 1.0 + K/(m*c_light*c_light + 1e-12)
            vfrac = np.sqrt(1.0 - 1.0/np.maximum(gamma*gamma, 1.0))
        vfrac = np.clip(vfrac, 0.0, 1.0)
        In = (I - I.min()) / (I.max() - I.min() + 1e-12)
        alpha = 0.35 + 0.65*In
        step = max(1, n_pts//20)
        xi, yi, zi = x[::step], y[::step], z[::step]
        Ii = power_intensity_ellipsoidal(xi, yi, zi, axes=(a,b,c))
        Inq = (Ii - Ii.min()) / (Ii.max() - Ii.min() + 1e-12)
        sizes = (18.0 * (1.0 + 0.6*Inq)).tolist()
        trajs.append({
            "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
            "vfrac": vfrac.tolist(), "alpha": alpha.tolist(),
            "qubits": {"x": xi.tolist(), "y": yi.tolist(), "z": zi.tolist(), "size": sizes}
        })

    return jsonify({
        "ellipsoid_lines": ell_lines,
        "power_lines": plines,
        "trajectories": trajs
    })

@app.get("/api/thruster")
def api_thruster():
    Q = float(request.args.get("Q", 0.2))
    R = float(request.args.get("R", 0.5))
    M_max = float(request.args.get("Mmax", 100.0))
    n = int(request.args.get("n", 200))
    x = np.linspace(0, 10, n)
    F = Q * np.sin(x)
    x_n = np.cos(x)
    x_np1 = np.roll(x_n, -1)
    M = R * M_max * np.ones_like(x)
    # Space schematic points
    sun = {"x": 5.0, "y": 50.0}
    wormhole_a = {"x": 3.0, "y": 20.0}
    wormhole_b = {"x": 8.0, "y": 80.0}
    ship = {"x": 7.5, "y": 75.0}
    return jsonify({
        "x": x.tolist(), "F": F.tolist(), "xn": x_n.tolist(), "xnp1": x_np1.tolist(), "M": M.tolist(),
        "sun": sun, "wormhole_a": wormhole_a, "wormhole_b": wormhole_b, "ship": ship
    })

@app.get("/api/timegraphs")
def api_timegraphs():
    np.random.seed(42)
    NUM = int(request.args.get("num", 400))
    R0_E = float(request.args.get("R0E", 24.0))
    R0_M = float(request.args.get("R0M", 30.0))
    AMP_E = float(request.args.get("AMPE", 2.0))
    AMP_M = float(request.args.get("AMPM", 3.0))
    TURNS = float(request.args.get("turns", 3.0))
    DELTA = float(request.args.get("delta", 0.15))
    PHASE = float(request.args.get("phase", 0.35*np.pi))
    ZSPAN = float(request.args.get("zspan", 25.0))

    earth = helical_on_cone(NUM, R0_E, AMP_E, TURNS, DELTA, 0.0, ZSPAN)
    moon  = helical_on_cone(NUM, R0_M, AMP_M, TURNS, DELTA, PHASE, ZSPAN)
    pairs = np.concatenate([earth, moon], axis=1)  # (N, 6)

    earth_dc = direction_change_indices(earth, 15.0)
    moon_dc  = direction_change_indices(moon,  15.0)

    # Invariance vs space: distance between end of seg i and start of seg i+1
    def seg_endpoints(points, dc_idx):
        if len(dc_idx) < 2:
            return []
        endpoints = []
        prev = 0
        for i in dc_idx + [len(points)-1]:
            endpoints.append(points[i])
            prev = i + 1
        return endpoints

    e_end = seg_endpoints(earth, earth_dc)
    inv_space = []
    if len(e_end) >= 2:
        for i in range(len(e_end)-1):
            inv_space.append(float(np.linalg.norm(e_end[i+1] - e_end[i])))

    # Invariance vs time: step-to-step spatial difference magnitude
    diffs = np.linalg.norm(np.diff(earth, axis=0), axis=1)
    inv_time = diffs.tolist()

    return jsonify({
        "earth": {"x": earth[:,0].tolist(), "y": earth[:,1].tolist(), "z": earth[:,2].tolist(), "dc": earth_dc},
        "moon":  {"x": moon[:,0].tolist(),  "y": moon[:,1].tolist(),  "z": moon[:,2].tolist(),  "dc": moon_dc},
        "pairs": pairs.tolist(),
        "inv_space": inv_space,
        "inv_time": inv_time
    })

@app.get("/api/ctc")
def api_ctc():
    # Parameters
    Gmu = float(request.args.get("Gmu", CFG.Gmu))
    beta = float(request.args.get("beta", CFG.beta))
    n_rays = int(request.args.get("n", CFG.n_rays))
    Lmin, Lmax = CFG.L_span
    L_vals = np.linspace(Lmin, Lmax, n_rays)
    alpha = 1.0 - 4.0*Gmu
    if alpha <= 0:
        alpha = 1e-3
    kappa = 0.0 if CFG.geodesic_type == "null" else -1.0

    fam = []
    for L in L_vals:
        t, r, ph, x, y, mask = integrate_geodesic(alpha, beta, CFG.E, L, CFG.r0, CFG.phi0, CFG.sgn_r, CFG.n_steps, CFG.dlam, kappa)
        fam.append({
            "L": float(L),
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
            "t": t[mask].tolist()
        })
    r_ctc = abs(beta)/alpha
    return jsonify({"family": fam, "r_ctc": float(r_ctc), "alpha": float(alpha), "beta": float(beta)})

if __name__ == "__main__":
    app.run(debug=True)