import smart
import dolfin
import pathlib
import logging

logger = logging.getLogger("smart")
logger.setLevel(logging.ERROR)

D_unit = smart.units.unit.um**2 / smart.units.unit.s
conc_unit_vol = smart.units.unit.molecule / smart.units.unit.um**3
conc_unit_surf = smart.units.unit.molecule / smart.units.unit.um**2
cyto_var = smart.model_assembly.Compartment("cyto", 3, smart.units.unit.um, 1)
ER_var = smart.model_assembly.Compartment("ER", 3, smart.units.unit.um, 2)
PM_var = smart.model_assembly.Compartment("PM", 2, smart.units.unit.um, 10)
ERM_var = smart.model_assembly.Compartment("ERM", 2, smart.units.unit.um, 12)
PM_var.specify_nonadjacency(["ERM", "ER"])
ERM_var.specify_nonadjacency(["PM"])
ER_var.specify_nonadjacency(["PM"])
A_var = smart.model_assembly.Species("A", 10, conc_unit_vol, 1.0, D_unit, "cyto")
B_var = smart.model_assembly.Species("B", 10, conc_unit_vol, 1.0, D_unit, "ER")
C_var = smart.model_assembly.Species("C", 10, conc_unit_surf, 1.0, D_unit, "PM")
D_var = smart.model_assembly.Species("D", 10, conc_unit_surf, 1.0, D_unit, "ERM")
kdecay_var = smart.model_assembly.Parameter("kdecay", 0.0, 1 / smart.units.unit.s)
r1_var = smart.model_assembly.Reaction(
    "r1", ["A"], [], param_map={"k": "kdecay"}, species_map={"sp": "A"}, eqn_f_str="k*sp"
)
r2_var = smart.model_assembly.Reaction(
    "r2", ["B"], [], param_map={"k": "kdecay"}, species_map={"sp": "B"}, eqn_f_str="k*sp"
)
r3_var = smart.model_assembly.Reaction(
    "r3", ["C"], [], param_map={"k": "kdecay"}, species_map={"sp": "C"}, eqn_f_str="k*sp"
)
r4_var = smart.model_assembly.Reaction(
    "r4", ["D"], [], param_map={"k": "kdecay"}, species_map={"sp": "D"}, eqn_f_str="k*sp"
)
cc = smart.model_assembly.CompartmentContainer()
cc.add([cyto_var, ER_var, PM_var, ERM_var])
sc = smart.model_assembly.SpeciesContainer()
sc.add([A_var, B_var, C_var, D_var])
pc = smart.model_assembly.ParameterContainer()
pc.add([kdecay_var])
rc = smart.model_assembly.ReactionContainer()
rc.add([r1_var, r2_var, r3_var, r4_var])
domain, facet_markers, cell_markers = smart.mesh_tools.create_cubes()
for face in dolfin.faces(domain):
    if face.midpoint().x() > dolfin.DOLFIN_EPS and facet_markers[face] == 10:
        facet_markers[face] = 0
mesh_folder = pathlib.Path("mesh")
mesh_folder.mkdir(exist_ok=True)
mesh_path = mesh_folder / "DemoCuboidsMesh.h5"
smart.mesh_tools.write_mesh(domain, facet_markers, cell_markers, filename=mesh_path)
parent_mesh = smart.mesh.ParentMesh(
    mesh_filename=str(mesh_path),
    mesh_filetype="hdf5",
    name="parent_mesh",
)
config_cur = smart.config.Config()
model_cur = smart.model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
config_cur.solver.update({"final_t": 0.1, "initial_dt": 0.1})
model_cur.initialize()
model_cur.monolithic_solve()
ui = model_cur.sc["A"].sol
for sp in ["A", "B", "C", "D"]:
    ui = model_cur.sc[sp].sol
    print(sp, ui.vector().norm("l2"), ui.function_space().mesh().num_cells(), len(ui.vector()))
