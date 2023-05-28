(sec:mathematics)=
# The SMART mathematical framework


SMART is designed to represent and solve coupled systems of nonlinear ordinary and partial differential equations, describing reactions and/or transport due to diffusion, convection or drift in multi-domain geometries including subdomains of co-dimension zero or one (volume-surface problems). This section defines the range and scope of SMART by describing its foundational mathematical modelling framework.

## Notation

- $\mathcal{M}$: index set for subdomains $\Omega^m$ ($m \in \mathcal{M}$).
- $\mathcal{Q}$: index set for the subsurfaces $\Gamma^q$ ($q \in \mathcal{Q}$).
- $\mathcal{I}$, $\mathcal{I}^m$, $\mathcal{I}^q$: index sets for the species, the species in $\Omega^m$ and species on $\Gamma^q$, respectively.


## Multi-domain and geometry representation

We consider an open topologically $D$-dimensional manifold $\Omega \subset \mathbb{R}^d$ for $D \leqslant d = 1, 2, 3$, and assume that $\Omega$ is partitioned into $|\mathcal{M}|$ open and disjoint \emph{subdomains} $\Omega^m \subset \mathbb{R}^d$:

```{math}
  \Omega = \bigcup_{m \in \mathcal{M}} \Omega^m ,
```

each with (internal or external) boundary $\partial \Omega^m$, and boundary $\partial \Omega$. The \emph{boundary normal} $\mathbf{n}^m$ on $\partial \Omega^m$ is the outward pointing normal vector field to the boundary. The interface $\Gamma^{mn}$ between subdomains $\Omega^m$ and $\Omega^n$ is defined as the intersection of the closure of the subdomains:
```{math}
  \Gamma^{mn} = \overline{\Omega^m} \cap \overline{\Omega^n} ,
```
for $m, n \in \mathcal{M}$, and may or may not be empty. Each interface $\Gamma^{mn}$ may be further partitioned into one or more \emph{subsurfaces}:
```{math}
    \Gamma^{mn} = \bigcup_{q \in \mathcal{Q}^{mn}} \Gamma^{q} .
```
We denote the total set of subsurfaces by $\mathcal{Q} = \cup_{m, n \in \mathcal{M}}$.

Moreover, let time $t \in [0, T]$ for $T > 0$.

## Coupled multi-domain reaction-transport equations

Different physical processes may occur on (sub)domains or on (sub)surfaces. For instance, in the context of computational neuroscience at the cellular level, different species and processes dominate in the dendrites and axons, and on the plasma membrane versus at post-synaptic densities. In general, SMART is designed to model different species coexisting, interacting and moving within a (sub)domain or (sub)surface, between (sub)domains and across (sub)surfaces.

### Domain species
Specifically, we define the set of species coexisting in $\Omega^m$ by $\mathcal{I}^m$ for each (sub)domain $\Omega^m$. The species $i \in \mathcal{I}^m$ with concentrations $u_i^m = u_i^m(x, t)$ for $x \in \Omega^m$ and $t \in (0, T]$ satisfy reaction-transport equations of the form
```{math}
\partial_t u_i^m + \mathcal{T}_i^m(u_i^m) - f_i^m(u_i^m) = 0 \qquad \text{ in } \Omega^m,
```
where $\partial_t$ is the time-derivative, $\mathcal{T}_i^m$ defines transport terms, and $f_i^m$ volume fluxes. In the case of transport by diffusion alone,
```{math}
    \mathcal{T}_i^m (u) = - \nabla \cdot ( D_i^m \nabla (u) ) ,
```
where $\nabla \cdot$ is the spatial divergence operator, $\nabla$ is the spatial gradient, and $D_i^m$ is the diffusion coefficient of species $i$ in subdomain $\Omega^m$, possibly heterogeneous i.e.~spatially varying and anisotropic~i.e.~tensor-valued.  We label $u^m = \{u_i^m\}_{i \in \mathcal{I}^m}$ for $m \in \mathcal{M}$. Remaining relative to $\Omega^m$, on (all) subsurfaces $\Gamma^{q} \subseteq \Gamma^{mn}$, we assume that the flux of species $i$ is governed by a relation of the form:
```{math}
:label: eq:robin
    D_i \nabla u_i^m \cdot n^m - R_i^{q} (u^m, u^n, v^q) = 0 \qquad \text{ on } \Gamma^{q} ,
```
where $R_i^q$ defines, typically non-linear, non-trivial, surface fluxes, and the surface concentrations $v^q$ are defined below.

### Surface species
The (sub)surface concentrations $v^q = \{ v_i^q \}_{i \in \mathcal{I}_q}$, entering in {eq}`eq:robin` above, are defined on $\Gamma^q \subseteq \Gamma^{mn}$, either as prescribed fields or via subsurface equations as follows: find $v^q = v^q(x, t)$ for $x \in \Gamma^q$, $t \in (0, T]$ such that
```{math}
  \partial_t v_i^q + \mathcal{T}_i^q(v_i^q ) - g_i^q ( u^m, u^n, v^q ) = 0 \qquad \text{ on } \Gamma^{q} .
```
where $g_i^q$ are surfaces fluxes for each species $i \in \mathcal{I}^q$. In the case of transport by surface diffusion, with surface diffusion coefficient $D_i^q$ for species $i$, surface gradient $\nabla_S$ and surface divergence $\nabla_S \cdot$, we have:
```{math}
    \mathcal{T}_i^q( v ) = - \nabla_S \cdot (D_i^q \nabla_S v ) .
```

### Boundary conditions
For any (sub)boundary $\Gamma^q \subset \partial \Omega^{m}$ such that $\Gamma^q \subseteq \partial \Omega$, zero flux boundary conditions are prescribed:
```{math}
    D_i^m u_i^m \cdot n^m = 0 .
```
This condition is also prescribed for any species on any (interior) subsurface $\Gamma^q$ with non-empty boundary.

### Initial conditions
Initial conditions are required for any $u_i^m$ for $m \in \mathcal{M}$, $i \in \mathcal{I}^m$, and $v_i^q$ for $q \in \mathcal{Q}$, $i \in \mathcal{I}^q$.
