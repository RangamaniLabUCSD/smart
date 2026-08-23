import dolfin as d
import os
import sympy as sym
import numpy as np
import pathlib
import gmsh  # must be imported before pyvista if dolfin is imported first
import sys

# Use the CH-enabled SMART tree (same as example1_CH copy.ipynb). Without this,
# system Python loads /usr/lib/python3/dist-packages/smart, which has no CH=.
sys.path.insert(0, "/home/shared/smartdir/smart-smart-withCH")

from smart import config, common, mesh, model, mesh_tools, visualization
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    SpeciesContainer,
    ParameterContainer,
    CompartmentContainer,
    ReactionContainer,
)
import logging
import atexit
from datetime import datetime

logger = logging.getLogger("smart")
logger.setLevel(logging.INFO)

# Aliases - base units
um = unit.um
molecule = unit.molecule
sec = unit.sec
dimensionless = unit.dimensionless
D_unit = um**2 / sec
flux_unit = molecule / (um * sec)
surf_unit = molecule / um**2
vol_unit = molecule / um**3
edge_unit = molecule / um

surf = Compartment("surf", 2, um, 10)

cc = CompartmentContainer()
cc.add([surf])

l0 = 1.0#np.sqrt(4*np.pi) # reference length scale
Shat = 30.0
phi0_X = 0.1
phi0_B = 0.1
sigma_s = Shat/l0**2
Lmax = (Shat**(3/2))/l0**3
Linit = 0.01*Lmax
print(Linit)
X = Species("X", phi0_X*sigma_s, surf_unit, 0.01, D_unit, "surf", CH=True, umax=sigma_s, A_hat = 0) 
B = Species("B", phi0_B*sigma_s, surf_unit, 0.01, D_unit, "surf", CH=True, umax=sigma_s, A_hat = 20.0)
L = Species("L", Linit, vol_unit, 1.0, D_unit, "vol", umax=Lmax)

sc = SpeciesContainer()
sc.add([L,X,B])

Kd = 7.5e-6 * (Shat**(3/2))/l0**3
koff_hat = 1.0 # 0.0 #0.6
kon_hat = koff_hat / Kd #0.9
tref = l0**2 / float(L.D)
kon = Parameter("kon", kon_hat/tref, 1/(vol_unit*sec))
# kon = Parameter("kon", 0.1*kon_hat*10/tref, surf_unit/sec)
koff = Parameter("koff", koff_hat/tref, 1/sec)
# Conversion of X to B
r1 = Reaction("r1", ["X","L"], ["B"],
              param_map={"kon": "kon", "koff": "koff"},
              eqn_f_str="kon*X*L - B*koff")

pc = ParameterContainer()
pc.add([kon, koff])
rc = ReactionContainer()
rc.add([r1])

useSpheroid = True
if useSpheroid:
    rOuter = [0.6849, 0.4365, 2.1896]
    rInner = [0.0,0.0,0.0]
    hEdge = 0.1
    domain, facet_markers, cell_markers = mesh_tools.create_ellipsoids(rOuter, rInner, hEdge=hEdge)
    # domain, facet_markers, cell_markers = mesh_tools.create_cubes(N=21)
    # for f in d.facets(domain):
    #     if f.midpoint().z() < 0.999:
    #         facet_markers[f] = 0
    vol = Compartment("vol", 3, um, 1) # SMART just needs to know its a 3d mesh
    cc.add(vol)
    # mesh_file = pathlib.Path("spheroid_ellipsoid_mesh.h5")
    mesh_file = pathlib.Path("spheroid_ellipsoid_mesh_new.h5")
    mesh_tools.write_mesh(domain, facet_markers, cell_markers, filename=mesh_file)
else:
    # define dimensions of domain
    Shat = 200
    x_size = np.sqrt(Shat/B.umax)
    y_size = np.sqrt(Shat/B.umax)
    # Create mesh
    m = 40
    n = int(x_size/y_size)*m
    rect_mesh = d.RectangleMesh(d.Point(0.0, 0.0), d.Point(x_size, y_size), n, m)
    mf2 = d.MeshFunction("size_t", rect_mesh, 2, 10)
    mf1 = d.MeshFunction("size_t", rect_mesh, 1, 0)
    class OuterEdge(d.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    outerEdge = OuterEdge()
    outerEdge.mark(mf1, 3)
    mesh_folder = pathlib.Path("rect_mesh")
    mesh_folder.mkdir(exist_ok=True)
    mesh_file = mesh_folder / "rect_mesh.h5"
    mesh_tools.write_mesh(rect_mesh, mf1, mf2, mesh_file)

parent_mesh = mesh.ParentMesh(
    mesh_filename=str(mesh_file),
    mesh_filetype="hdf5",
    name="parent_mesh",
)
config_cur = config.Config()
config_cur.flags.update({"allow_unused_components": True})
config_cur.solver.update(
    {
        "final_t": 1000.0,
        "initial_dt": 0.001,
        "time_precision": 8,
        "attempt_timestep_restart_on_divergence": True,
    }
)

# Result folder + run log (tee stdout/stderr so print() and SMART logs are saved).
# Installed before initialize() so JIT/solver setup is captured too.
SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
# SCRATCH_DIRECTORY was for server runs (e.g. SLURM scratch). Uncomment to use it instead of SCRIPT_DIR.
# SCRATCH_DIRECTORY = pathlib.Path(os.environ.get("SCRATCH_DIRECTORY", "/root/scratch"))


def _default_result_folder():
    name = (
        f"S{Shat}_AX{X.A_hat}_Ab{B.A_hat}_"
        f"phi0X{phi0_X}_phi0B{phi0_B}_phi0L{Linit}_"
        f"D_X{X.D}_D_B{B.D}_D_L{L.D}_"
        f"kon{kon_hat}_koff{koff_hat}_l0{l0}"
    )
    if useSpheroid:
        name = (
            f"a{rOuter[0]}_b{rOuter[1]}_c{rOuter[2]}_hEdge{hEdge}_" + name
        )
    return SCRIPT_DIR / name / "results_reactRedo"


class _Tee:
    """Write to the terminal and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
        self.errors = getattr(streams[0], "errors", "strict")
        self.name = getattr(streams[-1], "name", "<tee>")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()


_result_base = os.environ.get("RESULT_BASE_DIR", "").strip()
result_folder = pathlib.Path(_result_base) if _result_base else _default_result_folder()
result_folder.mkdir(parents=True, exist_ok=True)
_log_path = result_folder / "run.log"
_log_file = open(_log_path, "a", encoding="utf-8")
atexit.register(_log_file.close)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"[log] writing stdout/stderr to {_log_path}", flush=True)
print(
    f"[log] start {datetime.now().isoformat(timespec='seconds')} "
    f"cwd={os.getcwd()} Linit={Linit} Shat={Shat} A_hat_B={B.A_hat}",
    flush=True,
)

model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
model_cur.initialize()
model_cur.to_pickle(str(result_folder / "model_cur.pkl"))


import petsc4py.PETSc as PETSc
# most of these were originally set in smart.model and altered here to tailor this problem
if config_cur.solver['use_snes']:
    model_cur.solver.setType('newtonls')
    opts = PETSc.Options()
    opts['snes_linesearch_type'] = 'basic'
    model_cur.solver.setFromOptions()
    # set number of failed solves (tailored to this problem)
    model_cur.solver.setMaxKSPFailures(200)
    model_cur.solver.setMaxLinearSolveFailures(200)
    # relax solver tolerances (tailored to this problem)
    rtol_scale = 1.5
    atol_scale = 1
    model_cur.solver.setTolerances(1e-6*rtol_scale, 1e-6*atol_scale, 1e-20, 50)
    ksp_rtol_scale = 1.5
    ksp_atol_scale = 1
    ksp_maxits = 1e5  # default 1e4
    model_cur.solver.ksp.setTolerances(1e-4*ksp_rtol_scale, 1e-6*ksp_atol_scale, 1e6, ksp_maxits)



# Two-constraint mass corrector (same logic as TSCC/260816/S10_A20.py).
# SMART stores surface densities u = phi * umax; simplex is applied in phi-space.
# Ligand budget is dimensional: int L dv + int B ds (no S^{-1/2} prefactor).
epsilon = 1e-5
xi_secant_max_iter = 500
secant_tol = 1e-10

Xfunc = model_cur.sc["X"].u["u"]
Xdof = model_cur.sc["X"].dof_map
Bfunc = model_cur.sc["B"].u["u"]
Bdof = model_cur.sc["B"].dof_map
Lfunc = model_cur.sc["L"].u["u"]
Ldof = model_cur.sc["L"].dof_map
dx_pm = d.Measure("dx", model_cur.cc["surf"].dolfin_mesh)
dx_vol = d.Measure("dx", model_cur.cc["vol"].dolfin_mesh)
umax_X = float(model_cur.sc["X"].umax)
umax_B = float(model_cur.sc["B"].umax)

X_ic = d.Function(model_cur.sc["X"].V)
B_ic = d.Function(model_cur.sc["B"].V)
L_ic = d.Function(model_cur.sc["L"].V)
d.assign(X_ic, model_cur.sc["X"].sol)
d.assign(B_ic, model_cur.sc["B"].sol)
d.assign(L_ic, model_cur.sc["L"].sol)

c_mass_tmp_X = d.Function(model_cur.sc["X"].V)
c_mass_tmp_B = d.Function(model_cur.sc["B"].V)
c_mass_tmp_L = d.Function(model_cur.sc["L"].V)
c_x_proj = d.Function(model_cur.sc["X"].V)
c_b_proj = d.Function(model_cur.sc["B"].V)
c_L_proj = d.Function(model_cur.sc["L"].V)


def _as_scalar_xi(val):
    return float(np.asarray(val).reshape(-1)[0])


def assign_from_sub(subfunc, func, dofmap):
    fullvec = func.vector()[:]
    subvec = subfunc.vector()[:]
    fullvec[dofmap] = subvec
    func.vector().set_local(fullvec)
    func.vector().apply("insert")


def _project_simplex_XB(x_vals, b_vals, eps):
    """Nodewise Euclidean projection of (phi_X, phi_B) onto
    S_eps = {phi_B >= eps, phi_X >= eps, phi_B + phi_X <= 1 - eps}.
    """
    x = np.array(x_vals, dtype=float, copy=True)
    b = np.array(b_vals, dtype=float, copy=True)
    cap = 1.0 - float(eps)
    over = (b + x) > cap
    if np.any(over):
        t = 0.5 * ((b[over] + x[over]) - cap)
        b[over] = b[over] - t
        x[over] = x[over] - t
    lo_b = b < eps
    if np.any(lo_b):
        b[lo_b] = eps
        x[lo_b] = np.clip(x[lo_b], eps, cap - eps)
    lo_x = x < eps
    if np.any(lo_x):
        x[lo_x] = eps
        b[lo_x] = np.clip(b[lo_x], eps, cap - eps)
    return x, b


def _set_projected_XB(c_tmp_X, c_tmp_B, x_src, b_src, xi1_scalar, xi2_scalar, dt_val, eps):
    """X <- X_tilde + dt*xi_1,
    B <- B_tilde + dt*(xi_1 + xi_2),
    then jointly project fractions onto S_eps.
    """
    d.assign(c_tmp_X, x_src)
    d.assign(c_tmp_B, b_src)
    xv = np.asarray(c_tmp_X.vector()[:], dtype=float)
    bv = np.asarray(c_tmp_B.vector()[:], dtype=float)
    xi1 = _as_scalar_xi(xi1_scalar)
    xi2 = _as_scalar_xi(xi2_scalar)
    xv = xv + float(dt_val) * xi1
    bv = bv + float(dt_val) * (xi1 + xi2)
    phi_x, phi_b = _project_simplex_XB(xv / umax_X, bv / umax_B, float(eps))
    c_tmp_X.vector().set_local(phi_x * umax_X)
    c_tmp_X.vector().apply("insert")
    c_tmp_B.vector().set_local(phi_b * umax_B)
    c_tmp_B.vector().apply("insert")


def _set_clipped_sum_c_tmp_vol_scalar(c_tmp, c_src, xi2_scalar, dt_val):
    """Bulk L <- clip(L_tilde + dt*xi_2) into [eps*umax, (1-eps)*umax]."""
    umax_L = float(model_cur.sc["L"].umax)
    lo = umax_L * float(epsilon)
    hi = umax_L * (1.0 - float(epsilon))
    d.assign(c_tmp, c_src)
    c_vals = np.asarray(c_tmp.vector()[:], dtype=float)
    y = np.clip(c_vals + float(dt_val) * _as_scalar_xi(xi2_scalar), float(lo), float(hi))
    c_tmp.vector().set_local(y)
    c_tmp.vector().apply("insert")


def F_mass(xi_arg1, xi_arg2, dt_val):
    """F(xi_1,xi_2) = int(P_X + P_B) - int(X_ic + B_ic)."""
    _set_projected_XB(
        c_mass_tmp_X, c_mass_tmp_B,
        model_cur.sc["X"].sol, model_cur.sc["B"].sol,
        xi_arg1, xi_arg2, dt_val, float(epsilon),
    )
    form = (c_mass_tmp_X + c_mass_tmp_B - X_ic - B_ic) * dx_pm
    return float(d.assemble_mixed(form))


def G_mass(xi_arg1, xi_arg2, dt_val):
    """G(xi_1,xi_2) = int(P_B - B_ic) ds + int(P_L - L_ic) dv."""
    _set_projected_XB(
        c_mass_tmp_X, c_mass_tmp_B,
        model_cur.sc["X"].sol, model_cur.sc["B"].sol,
        xi_arg1, xi_arg2, dt_val, float(epsilon),
    )
    _set_clipped_sum_c_tmp_vol_scalar(
        c_mass_tmp_L, model_cur.sc["L"].sol, xi_arg2, dt_val
    )
    form_B = (c_mass_tmp_B - B_ic) * dx_pm
    form_L = (c_mass_tmp_L - L_ic) * dx_vol
    return float(d.assemble_mixed(form_B)) + float(d.assemble_mixed(form_L))


def J_abcd(xi1_prev, xi2_prev, xi1_curr, xi2_curr, dt_val):
    x1p, x2p = _as_scalar_xi(xi1_prev), _as_scalar_xi(xi2_prev)
    x1c, x2c = _as_scalar_xi(xi1_curr), _as_scalar_xi(xi2_curr)
    a_k = (F_mass(x1c, x2c, dt_val) - F_mass(x1p, x2c, dt_val)) / (x1c - x1p)
    b_k = (F_mass(x1c, x2c, dt_val) - F_mass(x1c, x2p, dt_val)) / (x2c - x2p)
    c_k = (G_mass(x1c, x2c, dt_val) - G_mass(x1p, x2c, dt_val)) / (x1c - x1p)
    d_k = (G_mass(x1c, x2c, dt_val) - G_mass(x1c, x2p, dt_val)) / (x2c - x2p)
    return a_k, b_k, c_k, d_k


def _vertex_min_max_mean(f):
    vals = np.asarray(f.compute_vertex_values(), dtype=float)
    return float(np.min(vals)), float(np.max(vals)), float(np.mean(vals))


def _integrals_XB_L():
    int_X = float(d.assemble_mixed(model_cur.sc["X"].sol * dx_pm))
    int_B = float(d.assemble_mixed(model_cur.sc["B"].sol * dx_pm))
    int_L = float(d.assemble_mixed(model_cur.sc["L"].sol * dx_vol))
    return int_X, int_B, int_L


def _surface_sum_phi_stats():
    """Min / max / mean of (phi_X + phi_B) at P1 vertices (fraction space)."""
    xv = np.asarray(model_cur.sc["X"].sol.compute_vertex_values(), dtype=float) / umax_X
    bv = np.asarray(model_cur.sc["B"].sol.compute_vertex_values(), dtype=float) / umax_B
    s = xv + bv
    return float(np.min(s)), float(np.max(s)), float(np.mean(s))


def dt_scale_from_max_newton(max_newton_its):
    """Map max SNES Newton iteration count to dt scale (same bins as adjust_dt.py)."""
    n = int(max_newton_its)
    if n in (0, 1):
        return 1.05
    if n in (2, 3):
        return 1.01
    if n in (4, 5):
        return 1.005
    if n in (6, 7, 8, 9, 10):
        return 0.9
    if n in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
        return 0.8
    else:
        return 0.5



# Write initial condition(s) to file
results = dict()
for species_name, species in model_cur.sc.items:
    results[species_name] = d.XDMFFile(
        model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
    )
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

# Set loglevel to warning in order not to pollute notebook output
logger.setLevel(logging.WARNING)

int_X0, int_B0, int_L0 = _integrals_XB_L()
int_XB0 = int_X0 + int_B0
int_LB0 = int_L0 + int_B0
mn_X, mx_X, av_X = _vertex_min_max_mean(model_cur.sc["X"].sol)
mn_B, mx_B, av_B = _vertex_min_max_mean(model_cur.sc["B"].sol)
mn_L, mx_L, av_L = _vertex_min_max_mean(model_cur.sc["L"].sol)
print(
    f"[IC] int X={int_X0:.12g}  int B={int_B0:.12g}  int(X+B)={int_XB0:.12g}  "
    f"int L={int_L0:.12g}  int L + int B={int_LB0:.12g}",
    flush=True,
)
print(
    f"[IC] X min/max/mean={mn_X:.6g}/{mx_X:.6g}/{av_X:.6g}  "
    f"B min/max/mean={mn_B:.6g}/{mx_B:.6g}/{av_B:.6g}  "
    f"L min/max/mean={mn_L:.6g}/{mx_L:.6g}/{av_L:.6g}",
    flush=True,
)

tvec = [float(model_cur.t)]
total_L_list = [int_L0]
total_X_list = [int_X0]
total_B_list = [int_B0]
total_XB_list = [int_XB0]
min_X_list, max_X_list, mean_X_list = [mn_X], [mx_X], [av_X]
min_B_list, max_B_list, mean_B_list = [mn_B], [mx_B], [av_B]
_, mx_xb0, av_xb0 = _surface_sum_phi_stats()
max_XB_list, mean_XB_list = [mx_xb0], [av_xb0]
mass_cons_vec = [[int_LB0, int_XB0]]
np.savetxt(str(result_folder / "mass_cons_vec.txt"), mass_cons_vec)


def _save_monitoring_timeseries():
    np.savez_compressed(
        result_folder / "monitoring_timeseries.npz",
        t=np.asarray(tvec, dtype=float),
        total_L=np.asarray(total_L_list, dtype=float),
        total_X=np.asarray(total_X_list, dtype=float),
        total_B=np.asarray(total_B_list, dtype=float),
        total_XB=np.asarray(total_XB_list, dtype=float),
        max_X=np.asarray(max_X_list, dtype=float),
        min_X=np.asarray(min_X_list, dtype=float),
        mean_X=np.asarray(mean_X_list, dtype=float),
        max_B=np.asarray(max_B_list, dtype=float),
        min_B=np.asarray(min_B_list, dtype=float),
        mean_B=np.asarray(mean_B_list, dtype=float),
        max_XB=np.asarray(max_XB_list, dtype=float),
        mean_XB=np.asarray(mean_XB_list, dtype=float),
        epsilon=np.asarray([float(epsilon)], dtype=float),
        S_hat=np.asarray([float(Shat)], dtype=float),
    )


_save_monitoring_timeseries()
print(f"[monitor] initial timeseries written t={float(model_cur.t):.8g}", flush=True)

dt_max_base = 0.02

# Solve
while True:
    print(f"Time is {model_cur.t}", flush=True)
    model_cur.monolithic_solve()
    dt_step = float(model_cur.dt)
    n_newton = model_cur.idx_nl[-1]
    print(f"Number of Newton its: {n_newton}", flush=True)

    int_X_pred, int_B_pred, int_L_pred = _integrals_XB_L()
    print(
        f"[pred-mass] int(X+B)={int_X_pred + int_B_pred:.12g} "
        f"(IC {int_XB0:.12g}, diff={int_X_pred + int_B_pred - int_XB0:.6e}) | "
        f"int L + int B={int_L_pred + int_B_pred:.12g} "
        f"(IC {int_LB0:.12g}, diff={int_L_pred + int_B_pred - int_LB0:.6e})",
        flush=True,
    )

    # Two-scalar secant on (xi_1, xi_2); initial guesses (0,0) and (-dt,-dt).
    xi_secant_iter = 0
    xi_guess_prev = [0.0, 0.0]
    xi_guess = [-dt_step, -dt_step]
    xi_guess_next = None
    a_k, b_k, c_k, d_k = J_abcd(
        xi_guess_prev[0], xi_guess_prev[1], xi_guess[0], xi_guess[1], dt_step
    )
    F1 = F_mass(xi_guess[0], xi_guess[1], dt_step)
    G1 = G_mass(xi_guess[0], xi_guess[1], dt_step)

    while xi_guess_next is None or (abs(F1) > secant_tol or abs(G1) > secant_tol):
        xi_secant_iter += 1
        if xi_secant_iter > xi_secant_max_iter:
            print(
                f"[secant] hit max iter={xi_secant_max_iter} F={F1:.6e} G={G1:.6e}",
                flush=True,
            )
            break
        denom = a_k * d_k - b_k * c_k
        if abs(denom) < 1e-30:
            print("[secant] singular Jacobian, stopping", flush=True)
            break
        f1 = F_mass(xi_guess[0], xi_guess[1], dt_step)
        g1 = G_mass(xi_guess[0], xi_guess[1], dt_step)
        xi_guess_next = [
            xi_guess[0] - (d_k * f1 - b_k * g1) / denom,
            xi_guess[1] - (a_k * g1 - c_k * f1) / denom,
        ]
        xi_guess_prev = [float(xi_guess[0]), float(xi_guess[1])]
        xi_guess = [float(xi_guess_next[0]), float(xi_guess_next[1])]
        a_k, b_k, c_k, d_k = J_abcd(
            xi_guess_prev[0], xi_guess_prev[1], xi_guess[0], xi_guess[1], dt_step
        )
        F1 = F_mass(xi_guess[0], xi_guess[1], dt_step)
        G1 = G_mass(xi_guess[0], xi_guess[1], dt_step)

    xi1_final, xi2_final = float(xi_guess[0]), float(xi_guess[1])
    print(
        f"Secant approach converged in {xi_secant_iter} iterations  "
        f"F={F1:.6e} G={G1:.6e} xi1={xi1_final:.6g} xi2={xi2_final:.6g}",
        flush=True,
    )

    _set_projected_XB(
        c_x_proj, c_b_proj,
        model_cur.sc["X"].sol, model_cur.sc["B"].sol,
        xi1_final, xi2_final, dt_step, float(epsilon),
    )
    _set_clipped_sum_c_tmp_vol_scalar(
        c_L_proj, model_cur.sc["L"].sol, xi2_final, dt_step
    )
    assign_from_sub(c_x_proj, Xfunc, Xdof)
    assign_from_sub(c_b_proj, Bfunc, Bdof)
    assign_from_sub(c_L_proj, Lfunc, Ldof)

    if len(model_cur.problem.global_sizes) == 1:
        model_cur._ubackend = model_cur.u["u"]._functions[0].vector().vec().copy()
    else:
        model_cur._ubackend = PETSc.Vec().createNest(
            [usub.vector().vec().copy() for usub in model_cur.u["u"]._functions],
            comm=model_cur.mpi_comm_world,
        )
    model_cur._init_4_4_get_species_u_v_V_dofmaps()
    model_cur._init_4_5_name_functions()
    model_cur.update_solution()

    int_X, int_B, int_L = _integrals_XB_L()
    int_XB = int_X + int_B
    int_LB = int_L + int_B
    mass_cons_vec.append([int_LB, int_XB])
    np.savetxt(str(result_folder / "mass_cons_vec.txt"), mass_cons_vec)
    mn_X, mx_X, av_X = _vertex_min_max_mean(model_cur.sc["X"].sol)
    mn_B, mx_B, av_B = _vertex_min_max_mean(model_cur.sc["B"].sol)
    mn_L, mx_L, av_L = _vertex_min_max_mean(model_cur.sc["L"].sol)
    print(
        f"[corr-mass] int(X+B)={int_XB:.12g} (diff IC {int_XB - int_XB0:.6e}) | "
        f"int L + int B={int_LB:.12g} (diff IC {int_LB - int_LB0:.6e})",
        flush=True,
    )
    print(
        f"Min/max B={mn_B:.6g}/{mx_B:.6g}  X={mn_X:.6g}/{mx_X:.6g}  L={mn_L:.6g}/{mx_L:.6g}",
        flush=True,
    )

    mn_xb, mx_xb, av_xb = _surface_sum_phi_stats()
    print(f"[interface] max(phi_X+phi_B)={mx_xb:.6g} mean={av_xb:.6g}", flush=True)

    dtFactor = dt_scale_from_max_newton(n_newton)
    dt_next = min(dt_step * dtFactor, dt_max_base)
    if dt_next < 1e-5:
        dt_next = 1e-5
    model_cur.set_dt(dt_next)
    print(f"[dt] next dt={float(model_cur.dt):.6g} (scale={dtFactor})", flush=True)

    if np.mod(model_cur.idx, 100) == 0:
        for species_name, species in model_cur.sc.items:
            results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
        tvec.append(float(model_cur.t))
        total_L_list.append(int_L)
        total_X_list.append(int_X)
        total_B_list.append(int_B)
        total_XB_list.append(int_XB)
        min_X_list.append(mn_X)
        max_X_list.append(mx_X)
        mean_X_list.append(av_X)
        min_B_list.append(mn_B)
        max_B_list.append(mx_B)
        mean_B_list.append(av_B)
        max_XB_list.append(mx_xb)
        mean_XB_list.append(av_xb)
        _save_monitoring_timeseries()
        print(
            f"[monitor] timeseries updated t={float(model_cur.t):.8g} step={model_cur.idx}",
            flush=True,
        )

    if model_cur.t >= model_cur.final_t:
        break
