FROM quay.io/fenicsproject/dev

USER root

# Install some additional packages on top of standard FEniCS dev
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install curl vim htop bpython && \
    apt-get clean && \
    python3 -m pip install meshio && \
    python3 -m pip install gmsh && \
    python3 -m pip install bpython && \
    python3 -m pip install fenics-stubs && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
