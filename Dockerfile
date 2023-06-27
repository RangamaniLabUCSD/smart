FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21 as smart_base

ENV PYVISTA_JUPYTER_BACKEND="panel"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[test,examples]"
RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in amd64) \
    python3 -m pip install ".[pyvista]" ;; \
    esac;

RUN python3 -m pip install pre-commit

# Jupyter-lab images for examples
FROM smart_base as smart_lab
EXPOSE 8888/tcp
ENTRYPOINT [ "jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]
