(sec:numerics)=
# The SMART numerical solution algorithms


SMART solves the multi-domain reaction-transport equations outline in
{ref}`sec:mathematics` via finite difference discretizations in time, a finite element discretizations in space, and either a *monolithic* or *iterative* approach to address the system couplings. We describe the monolithic approach here, considering linear diffusion-only transport relations for concreteness and to illustrate linear-nonlinear strategies.

\subsection{Monolithic solution of the multi-domain reaction-diffusion equations}

## Variational formulation
Introduce the Sobolev spaces $H^1(\Omega^m)$, $m \in \mathcal{M}$ of square-integrable functions on $\Omega^m$ with square-integrable weak derivatives, and analogously for $H^1(\Gamma^q)$, $q \in Q$, as well as the vector function spaces $H^1(\Omega^m, \mathbb{R}^d) = H^1(\Omega)^d$. For solving the multi-domain reaction-diffusion equations, we introduce the two product spaces $U$ and $V$ consisting of (sub)domain fields and (sub)surface fields respectively:
```{math}
    U = \bigotimes_{m \in \mathcal{M}} H^1(\Omega^m; \mathbb{R}^{|\mathcal{I}^m|}), \quad
    V = \bigotimes_{q \in \mathcal{Q}} H^1(\Gamma^q; \mathbb{R}^{|\mathcal{I}^q|}) .
```
To represent the solution fields $u, v$ comprised of the separate (sub)domain and (sub)surface components, we label
```{math}
    u
    = \{ u^m \}_{m \in \mathcal{M}}
    = \{ \{u_i^m \}_{i \in \mathcal{I}^m} \}_{m \in \mathcal{M}},
    \quad
    v
    = \{ v^q \}_{q \in \mathcal{Q}}
    = \{ \{ v_i^q \}_{i \in \mathcal{I}^q} \}_{q \in \mathcal{Q}} .
```

By standard techniques (integrating by test functions and integrating by parts), we may rephrase the multi-domain reaction-diffusion equations in variational form: find $u \in U$ and $v \in V$ such that for all $\phi \in U$ and $\psi \in V$:
```{math}
    F(u, v; \phi) + G(u, v; \psi) = 0,
```
where both forms $F$ and $G$ are composed of sums over domains or surfaces and species:
```{math}
    F(u, v; \phi) = \sum_{m \in \mathcal{M}} \sum_{i \in \mathcal{I}^m} F_i^m(u, v; \phi_i^m), \qquad
    G(u, v; \psi) = \sum_{q \in \mathcal{Q}} \sum_{i \in \mathcal{I}^q} G_i^q(u, v; \psi_i^q) .
```
Furthermore, with the $L^2(O)$-inner product over any given domain $O \subset \Omega$ defined as
$$
    \langle a, b \rangle_{O} = \int_{O} a \cdot b \, \mathrm{d}x ,
$$
we have defined
```{math}
    F_i^m(u, v, \phi_i^m)
    = \langle \partial_t u_i^m, \phi_i^m \rangle_{\Omega^m}
    + \langle D_i^m \nabla u_i^m, \nabla \phi_i^m \rangle_{\Omega^m}
    - \langle f_i^m(u^m), \phi_i^m\rangle_{\Omega^m}
    - \sum_{q \in \mathcal{Q}^{mn}} \langle R_i^q(u^m, u^n, v^q), \phi_i^m\rangle_{\Gamma^q}
```
and, for any $q \in \mathcal{Q}^{mn}$ for given $\Omega^m$ and $\Omega^n$ interfacing $\Omega^m$ via $\Gamma^q$, finally:
```{math}
    G_i^q(u, v, \psi_i^q)
    = \langle \partial_t v^q, \psi_i^q \rangle_{\Gamma^q}
     + \langle D_i^q \nabla_S v_i^q, \nabla_S \psi_i^q \rangle_{\Gamma^q}
     - \langle g_i^q(u^m, u^n, v^q), \psi_i^q \rangle_{\Gamma^q} .
```

### Discretization in time and space
We now discretize in time via an implicit (first-order) Euler scheme with time steps $0 = t_0 < t_1 < \ldots < t_{n-1} < t_n = T$ with timestep $\tau_n = t_n - t_{n-1}$. To discretize in space, we consider a conforming finite element mesh $\mathcal{T}$ defined relative to the parcellation of $\Omega$ into subdomains, surfaces and subsurfaces, such that $\mathcal{T}^m \subseteq \mathcal{T}$ defines a submesh of $\Omega^m$ for $m \in \mathcal{M}$, and $\mathcal{G}^q$ defines a (co-dimension  one) mesh of $\Gamma^q$ for $q \in \mathcal{Q}$. We define the finite element spaces of continuous piecewise linears $\mathcal{P}_1$ defined relative to each (sub)mesh, and define the product spaces
```{math}
    U_h = \bigotimes_{m \in \mathcal{M}} \mathcal{P}_1^{|\mathcal{I}_m|}(\mathcal{T}^m), \qquad
    V_h = \bigotimes_{q \in \mathcal{Q}} \mathcal{P}_1^{|\mathcal{I}_q|}(\mathcal{G}^q) .
```
For each time step $t_i$, given discrete solutions $u_h^{-}$ and $v_h^{-}$ at the previous time $t_{i-1}$, we then solve the nonlinear, coupled time-discrete problem: find $u_h \in U_h$ and $v_h \in V_h$
```{math}
    \label{eq:nonlinear}
    H(u_h, v_h) = F_{\tau_n}(u_h, v_h, \phi) + G_{\tau_n}(u_h, v_h, \psi) = 0
```
for all $\phi \in U_h$, $\psi \in V_h$, where $F_{\tau_n}$ and $G_{\tau_n}$ are defined by the forms $F$ and $G$ after implicit Euler time-discretization and depend on the given $u_h^{-}$ and $v_h^{-}$.

### Nonlinear solution algorithms

By default, the monolithic nonlinear discrete system {eq}`eq:nonlinear` is solved by Newton-Raphson iteration with a symbolically derived discrete Jacobian.
