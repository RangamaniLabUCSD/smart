FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install "."
