```@meta
CurrentModule = GeometricBrackets
```

# Vlasov

The continuum Vlasov equation on a two-dimensional phase space ``\Omega = \Omega_x \times \Omega_v``
is a Lie-Poisson Hamiltonian system with respect to the canonical phase-space bracket
``\{a, b\}_{(x, v)} = \partial_x a \, \partial_v b - \partial_v a \, \partial_x b``:

```math
\{F, G\}_{\mathrm{LP}}[f] = \int_\Omega f(x, v) \,
\left\{ \frac{\delta F}{\delta f}, \frac{\delta G}{\delta f} \right\}_{(x, v)} dx \, dv .
```

## Why direct Eulerian projections fail the Jacobi identity

Projecting ``f_h = \sum_{m = 1}^N c_m \Phi_m`` directly onto a finite-dimensional spline,
spectral or finite-difference basis produces a state-linear bracket
``\mathbb{J}_{ij}(c) = \sum_m c_{ij}^m c_m`` with tensor [`vlasov_galerkin_tensor`](@ref).
By the structure-constant theorem, ``\mathbb{J}(c)`` satisfies the Jacobi identity if and
only if ``c_{ij}^m`` are the structure constants of a finite-dimensional Lie algebra.
Polynomial splines, truncated Fourier series and the nine-point [`Arakawa`](@ref) stencil do
not close under ``\{\cdot, \cdot\}_{(x, v)}``, so their [`structure_constant_residual`](@ref)
and [`jacobi_residual`](@ref) remain ``O(1)`` under mesh refinement. Nor can a pointwise
change of variables ``f = \Phi(\bar{f})`` flatten the bracket in two phase-space dimensions,
because ``\{h A, h B\}_{(x, v)} \neq h^2 \{A, B\}_{(x, v)}``.

## Four non-particle finite-dimensional Poisson brackets

This module provides four distinct non-particle routes that satisfy the Jacobi identity
exactly at every finite dimension:

1. **Time-reversal parity splitting** ([`velocity_parity_projectors`](@ref),
   [`parity_vlasov_blocks`](@ref), [`parity_vlasov_bracket`](@ref),
   [`dlra_parity_vlasov_bracket`](@ref)): under velocity reflection
   ``\mathcal{T}_v(x, v) = (x, L_v - v)``, the canonical bracket ``\{\cdot, \cdot\}_{(x, v)}``
   has odd parity. At any velocity-even equilibrium ``f_0(x, L_v - v) = f_0(x, v)``, the
   ``v``-even subspace (carrying charge density ``\rho``) and ``v``-odd subspace (carrying
   current density ``j``) are Lagrangian subspaces of the frozen Lie-Poisson operator
   ``\mathbb{J}(f_0)`` (`J_pp == 0`, `J_mm == 0`), producing a constant off-diagonal Poisson
   matrix ``\mathbb{J}_{\mathrm{par}}`` on spline, grid and ``O(r) \times O(r)``
   dynamic-low-rank Stiefel frames.
2. **Canonical Clebsch potentials** ([`canonical_vlasov_bracket`](@ref),
   [`clebsch_vlasov_tensor`](@ref), [`clebsch_vlasov_map`](@ref),
   [`clebsch_pushforward_matrix`](@ref)): representing the distribution through canonical
   potentials ``f_h = f_{0, h} + \Pi_h\{\alpha_h, \beta_h\}_{(x, v)}`` lifts the dynamics to
   ``(Q, P) \in (\mathbb{R}^{2N}, \mathbb{J}_{2N})``, where the bracket is constant and
   canonical while total mass ``\int_\Omega f_h \, dx \, dv`` is conserved identically by the
   B-spline partition of unity. The discrete Clebsch non-submersion theorem shows why the
   state must remain ``(Q, P)``: the pushforward bivector
   ``\Pi_{mn}(Q, P) = -Q^T [B_m, B_n] P`` descends to a Lie-Poisson bracket on ``c`` if and
   only if the slices ``B[m, :, :]`` close under the matrix commutator.
3. **Finite-mode von Neumann-Poisson quantisation on ``\mathfrak{u}(N)^*`` and spectral
   pushforwards** ([`un_hermitian_basis`](@ref), [`un_structure_constants`](@ref),
   [`weyl_vlasov_matrix`](@ref), [`spectral_pushforward_bracket`](@ref)): quantising the
   configuration torus into ``N = 2K + 1`` modes via compressed Weyl quantisation yields an
   exact Lie-Poisson bracket on ``\mathfrak{u}(N)^*`` that is exact on the inner mode block
   for symbols of degree ``\le 1`` in ``v`` and approximates Vlasov to ``O(\hbar^2)`` with
   leading Wigner dispersion. Pushing forward along ``V = \tfrac{1}{2}e^{2W} \succ 0`` yields
   the positive-definite Bogoliubov-Kubo-Mori logarithmic-mean Poisson bracket.
4. **Pointwise-flat velocity-moment / waterbag brackets**
   ([`vlasov_moment_flat_bracket`](@ref), [`vlasov_moment_direct_bracket`](@ref)):
   restricting to observables of degree ``\le 1`` in ``v`` closes into the semidirect product
   ``\mathrm{Vect}(\Omega_x) \ltimes C^\infty(\Omega_x)`` on ``(\rho, m)``. In one spatial
   dimension the pointwise chart ``(\rho_i, u_i = m_i / \rho_i)`` flattens the bracket into a
   constant skew-symmetric matrix for every discrete derivative ``\mathbb{K}_a`` (periodic or
   wall-bounded), so its pushforward ``\mathbb{J}(\rho, m) = D\Phi_N \mathbb{K}_2 D\Phi_N^T``
   satisfies the Jacobi identity identically, whereas direct collocation
   [`vlasov_moment_direct_bracket`](@ref) violates Jacobi at ``O(1)``.

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["equations/vlasov.jl"]
```
