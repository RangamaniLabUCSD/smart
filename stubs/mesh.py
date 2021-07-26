import dolfin as d

class Mesh(object):
    def __init__(self, mesh_filename=None, name='main_mesh', backend='dolfin'):
        self.mesh_filename  = mesh_filename
        self.name           = name
        self.backend        = backend
        if mesh_filename is not None:
            self.load_mesh()
    def load_mesh(self):
        if self.backend == 'dolfin':
            self.mesh = d.Mesh(self.mesh_filename)
            print(f"Mesh, {self.name}, successfully loaded from file: {self.mesh_filename}!")
        else: 
            raise ValueError(f"Backend {self.backend} is not supported yet")