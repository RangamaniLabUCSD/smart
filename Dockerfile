FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

ENV PYVISTA_JUPYTER_BACKEND="panel"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install "."
