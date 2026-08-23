#!/usr/bin/env python
# coding: utf-8

# # Example 2: Simple 2D cell signaling model
# Test alterations in mesh vs. Lagrangian mapping for different cases

import dolfin as d
import numpy as np
import pathlib
import logging

from smart import config, mesh, model, mesh_tools
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
from matplotlib import pyplot as plt

logger = logging.getLogger("smart")
logger.setLevel(logging.INFO)

um = unit.um
molecule = unit.molecule
sec = unit.sec
dimensionless = unit.dimensionless
D_unit = um**2 / sec
surf_unit = molecule / um**2
flux_unit = molecule / (um * sec)
edge_unit = molecule / um

# =============================================================================================
# Compartments
# =============================================================================================
# name, topological dimensionality, length scale units, marker value
Cyto = Compartment("Cyto", 2, um, 1)
PM = Compartment("PM", 1, um, 3)
cc = CompartmentContainer()
cc.add([Cyto, PM])

# =============================================================================================
# Species
# =============================================================================================
# name, initial concentration, concentration units, diffusion, diffusion units, compartment
A = Species("A", 1.0, surf_unit, 0.001, D_unit, "Cyto")
X = Species("X", 1.0, edge_unit, 0.001, D_unit, "PM")
B = Species("B", 0.0, edge_unit, 0.001, D_unit, "PM")
sc = SpeciesContainer()
sc.add([A, X, B])

# =============================================================================================
# Parameters and Reactions
# =============================================================================================

# Reaction of A and X to make B (Cyto-PM reaction)
kon = Parameter("kon", 1.0, 1 / (surf_unit * sec))
koff = Parameter("koff", 1.0, 1 / sec)
r1 = Reaction(
    "r1",
    ["A", "X"],
    ["B"],
    param_map={"on": "kon", "off": "koff"},
    species_map={"A": "A", "X": "X", "B": "B"},
)

pc = ParameterContainer()
pc.add([kon, koff])
rc = ReactionContainer()
rc.add([r1])

h_ellipse = 0.2
xrad = 1.0
yrad = 1.0
surf_tag = 1
edge_tag = 3
config_cur = config.Config()
config_cur.solver.update(
    {
        "final_t": 2.0,
        "initial_dt": 0.1,
        "time_precision": 6,
    }
)

# iterate over additional aspect ratios, enforcing the same ellipsoid area in all cases
aspect_ratios = [1, 1.5**2, 4, 9, 16]
l2_lagrange = []
l2_eul = []
Avecs = []

parent_folder = pathlib.Path("lagrange_testing")
parent_folder.mkdir(exist_ok=True)

for i in range(len(aspect_ratios)):

    udefs = []
    lagrangeSS = []
    refSS = []
    for lagrange in [False, True]:
        if lagrange:
            Cyto = Compartment(
                "Cyto",
                2,
                um,
                1,
                deform=(
                    f"x*{np.sqrt(aspect_ratios[i])-1}",
                    f"y*{1/np.sqrt(aspect_ratios[i]) - 1}",
                    "0",
                ),
            )
            PM = Compartment(
                "PM",
                1,
                um,
                3,
                deform=(
                    f"x*{np.sqrt(aspect_ratios[i])-1}",
                    f"y*{1/np.sqrt(aspect_ratios[i]) - 1}",
                    "0",
                ),
            )
        else:
            Cyto = Compartment("Cyto", 2, um, 1)
            PM = Compartment("PM", 1, um, 3)
        cc = CompartmentContainer()
        cc.add([Cyto, PM])

        # Create mesh
        if not lagrange:
            xrad = 1.0 * np.sqrt(aspect_ratios[i])
            yrad = 1.0 / np.sqrt(aspect_ratios[i])
            ellipse_mesh, mf1, mf2 = mesh_tools.create_ellipses(
                xrad,
                yrad,
                hEdge=h_ellipse,
                hInnerEdge=h_ellipse,
                outer_tag=surf_tag,
                outer_marker=edge_tag,
            )
            mesh_folder = parent_folder / f"ellipse_mesh_AR{aspect_ratios[i]}"
            mesh_folder.mkdir(exist_ok=True)
        else:
            mesh_folder = parent_folder / "ellipse_mesh_AR1"

        mesh_refinements = [0, 1, 2, 3, 4, 5, 6]
        for j in range(len(mesh_refinements)):
            mesh_file = mesh_folder / f"ellipse_mesh_{mesh_refinements[j]}.h5"
            if not lagrange:
                if mesh_refinements[j] > 0:
                    d.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
                    ellipse_mesh = d.adapt(ellipse_mesh)
                    mf2 = d.adapt(mf2, ellipse_mesh)
                    mf1 = d.adapt(mf1, ellipse_mesh)
                mesh_tools.write_mesh(ellipse_mesh, mf1, mf2, mesh_file)

            parent_mesh = mesh.ParentMesh(
                mesh_filename=str(mesh_file),
                mesh_filetype="hdf5",
                name="parent_mesh",
            )
            # init model
            model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
            model_cur.initialize()
            results = dict()
            result_folder = (
                parent_folder
                / f"resultsEllipse_AR{aspect_ratios[i]}_lagrange{lagrange}_{mesh_refinements[j]}"
            )
            result_folder.mkdir(exist_ok=True)
            for species_name, species in model_cur.sc.items:
                results[species_name] = d.XDMFFile(
                    model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
                )
                results[species_name].parameters["flush_output"] = True
                results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
            if lagrange:
                u1file = d.XDMFFile(model_cur.mpi_comm_world, str(result_folder / "u1var.xdmf"))
                u1file.parameters["flush_output"] = True
                u1file.write(cc["Cyto"].deform_func, model_cur.t)
                udef = cc["Cyto"].deform_func
                Fcur = d.Identity(3) + d.grad(udef)
                Jcur = d.det(Fcur)
            else:
                Jcur = d.Constant(1.0)
            avg_A = [A.initial_condition]
            dx = d.Measure("dx", domain=model_cur.cc["Cyto"].dolfin_mesh)
            volume = d.assemble_mixed(Jcur * dx)
            while True:
                # Solve the system
                model_cur.monolithic_solve()
                # Save results for post processing
                for species_name, species in model_cur.sc.items:
                    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
                if lagrange:
                    u1file.write(cc["Cyto"].deform_func, model_cur.t)
                int_val = d.assemble_mixed(Jcur * model_cur.sc["A"].u["u"] * dx)
                avg_A.append(int_val / volume)
                # End if we've passed the final time
                if model_cur.t >= model_cur.final_t:
                    break

            np.savetxt(str(result_folder / "avg_A.txt"), avg_A)
            Avecs.append(avg_A)

            if lagrange:
                plt.plot(
                    model_cur.tvec,
                    avg_A,
                    linestyle="dashed",
                    label=f"Aspect ratio = {aspect_ratios[i]}, Lagrange",
                )
                lagrangeSS.append(model_cur.sc["A"].sol.copy())
                udefs.append(udef)
            else:
                plt.plot(model_cur.tvec, avg_A, label=f"Aspect ratio = {aspect_ratios[i]}")
                refSS.append(model_cur.sc["A"].sol.copy())

    refFcn = refSS[-1]  # use the most refined reference (eulerian) mesh
    Vcur = refFcn.function_space()
    coords = Vcur.tabulate_dof_coordinates()
    dx = d.Measure("dx", domain=Vcur.mesh())
    l2_lagrangeCur = []
    l2_refCur = []
    for k in range(len(refSS) - 1):
        lagrange_fun = d.Function(Vcur)
        lagrange_vec = lagrange_fun.vector().get_local()
        eul_fun = d.Function(Vcur)
        eul_vec = eul_fun.vector().get_local()

        refSS[k].set_allow_extrapolation(True)
        Lmesh = lagrangeSS[k].function_space().mesh()
        d.ALE.move(Lmesh, udefs[k])
        Lmesh.bounding_box_tree().build(Lmesh)
        lagrangeSS[k].set_allow_extrapolation(True)
        for idx in range(len(lagrange_vec)):
            cur_coord = coords[idx]
            eul_vec[idx] = refSS[k](cur_coord)
            lagrange_vec[idx] = lagrangeSS[k](cur_coord)
        eul_fun.vector().set_local(eul_vec)
        eul_fun.vector().apply("insert")
        lagrange_fun.vector().set_local(lagrange_vec)
        lagrange_fun.vector().apply("insert")
        l2_lagrangeCur.append(np.sqrt(d.assemble((lagrange_fun - refFcn) ** 2 * dx)))
        l2_refCur.append(np.sqrt(d.assemble((eul_fun - refFcn) ** 2 * dx)))

    np.savetxt(str(parent_folder / f"l2_lagrange_AR{aspect_ratios[i]}.txt"), l2_lagrangeCur)
    np.savetxt(str(parent_folder / f"l2_eul_AR{aspect_ratios[i]}.txt"), l2_refCur)

plt.legend()
plt.show()
