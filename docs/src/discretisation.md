```@meta
CurrentModule = GeometricBrackets
```

# Discretisation

## Spaces

Two discrete function spaces are provided, and the choice between them is not free — it is
the discretisation each problem actually requires.

[`SplineSpace`](@ref) wraps the periodic B-spline basis of SimpleSplines. Being
``\mathcal{C}^{p-1}``, it can carry the third-derivative term of the KdV brackets: after one
integration by parts, ``-\int \phi_k \phi_l''' = \int \phi_k' \phi_l''``, degree ``p \ge 2``
suffices.

[`LagrangeSpace`](@ref) is a periodic nodal Lagrange finite element space. It is only
``\mathcal{C}^0``, but it is *nodal*, and that is what the Burgers construction needs: the
transformation ``\bar{u} = \sqrt{u}`` is applied coefficient by coefficient, which is only
justified when the coefficients are values at points.

Both answer the same small interface — `basis_values`, `quadrature_weights`,
`mass_matrix` — so every assembly downstream is written once. Those accessors are generics of
[SimpleSplines](https://github.com/JuliaDEC/SimpleSplines.jl), extended here rather than
redefined, so that the packages of the ecosystem share one function per accessor; their
reference documentation lives in that package.

## Assembly

Every matrix here is a weighted contraction of tabulated basis derivatives:

```math
\int_\Omega f(x) \, D^a \phi_k \, D^b \phi_l \, dx
  = \Phi_a \, \mathrm{diag}(f \odot w) \, \Phi_b^T .
```

That is [`weighted_matrix`](@ref); `mass_matrix`, [`stiffness_matrix`](@ref) and
[`derivative_matrix`](@ref) are the constant-coefficient cases.

## What the assemblies cost

Two structural facts do the work here, and both were arrived at by measuring rather than by
assumption — the first attempt at optimising this targeted the mass solve, which turned out
to be 0.1 % of a run.

**The basis tabulation is sparse.** ``\Phi`` is `N` by `n·nq`, but a basis function is
supported on `p+1` cells, so only `(p+1)·nq` entries per row are structurally nonzero.
Storing it densely makes every contraction ``\Phi \, \mathrm{diag}(fw) \, \Phi^T``
cost ``O(N^2 n n_q)`` where it should cost ``O(N p^2 n_q)``. At ``N = 384`` that alone was
the difference between 6.0 ms and 0.4 ms for one Hessian. Both spaces store it this way:
[`SplineSpace`](@ref) through `SimpleSplines.SplineQuadrature`, and [`LagrangeSpace`](@ref)
in its own assembly, where a dense table is 0.8 % full at `p = 2`, `ne = 192`.

**And nothing downstream may densify it.** [`AffineBracket`](@ref) holds the tabulation as a
field, and it used to call `Matrix` on the way in — which threw the sparsity away again for
the second KdV bracket and the first Camassa-Holm one, both of which contract it on every
Newton iteration. Keeping it sparse takes one [`jacobian`](@ref) of the second flow at
``N = 384`` from 7.2 ms to 1.3 ms, with `bracket_directional` alone 25 times faster. The two
``O(N^3)`` tensor routines — [`poisson_tensor`](@ref) and the Jacobiator's
[`poisson_derivative`](@ref) — densify locally and deliberately: they random-access the table
``N^3`` times instead of contracting it, and they are off the time loop by construction.

**The constant assemblies are memoised.** The mass, stiffness and derivative matrices and
``\int \phi_k' \phi_l''`` do not depend on the field, but the Hessian of ``H_1`` is
``\mathbb{K}^1 + 6\int \phi_i \phi_j u_h`` and was reassembling the stiffness matrix on
every Newton iteration of every step.

**The mass matrix is circulant on a uniform mesh** — the basis functions are then translates
of one cardinal spline — so it is diagonalised by the discrete Fourier transform and a solve
is two planned transforms and a pointwise division. On a graded or random mesh it is banded
modulo ``N`` but *not* circulant, and a sparse Cholesky is what is left. SimpleSplines
dispatches on the basis and, for a periodic basis, on its mesh; see its `MassOperator`. This
is the right representation of the operator, but it is worth being clear that it is not where
the time goes: at ``N = 384`` a mass solve is 0.001 ms against a 4.2 ms implicit step.

## Brackets

```math
\{A, B\}_d = \sum_{i,j} \frac{\partial A}{\partial \hat{u}_i} \,
             \mathbb{P}_{ij} (\hat{u}) \, \frac{\partial B}{\partial \hat{u}_j}
```

| type | antisymmetric | Jacobi |
|:--|:--|:--|
| [`ConstantBracket`](@ref) | yes | yes, automatically |
| [`GaugedBracket`](@ref) | yes | yes, by the theorem it is built on |
| [`MiuraBracket`](@ref) | yes | yes, being a pushforward |
| [`AffineBracket`](@ref) | yes | **no** |

## Why the skew-symmetrised form matters

Every bracket here is assembled in explicitly skew-symmetrised form. This is not cosmetic.

The unsymmetrised assembly of the second KdV bracket is the *same matrix* in exact
arithmetic — the term that differs is symmetric in ``k \leftrightarrow l`` and drops out of
the skew part. But it is antisymmetric only if the quadrature integrates a total derivative
of degree ``3p-1`` exactly. Written skew-symmetrised, antisymmetry holds **identically**, at
any number of quadrature points and on any mesh — and with it the exact conservation of the
generating Hamiltonian, which follows from antisymmetry alone:

```math
\dot{H} = \left( \frac{\partial H}{\partial \hat{u}} \right)^T \mathbb{P}
          \left( \frac{\partial H}{\partial \hat{u}} \right) = 0 .
```

Antisymmetry is thus the one property that has been made free. Accuracy has not: consistency
still needs degree ``3p-1``, and the mass still needs ``2p-1``.

## The Jacobi identity

[`AffineBracket`](@ref) does not satisfy it, and this is a result rather than a defect. The
operator ``4u\partial_x + 2u_x - \partial_x^3 = 2(u\partial_x + \partial_x u) -
\partial_x^3`` is the Virasoro Lie-Poisson structure, so the Jacobi identity would require
the discrete coefficients to close into a Lie algebra — a quadratic condition on the basis
that no choice of quadrature and no regrouping of the integrand can deliver. The normalised
[`jacobi_residual`](@ref) is of order one and *flat* under refinement:

```@example jac
using GeometricBrackets, Random
Random.seed!(1)
[jacobi_residual(kdv_bracket_2(SplineSpace(N, 3)), randn(N)) for N in (12, 16, 20, 24)]
```

[`MiuraBracket`](@ref) is the construction that does give an exactly Poisson second bracket,
by pushing the first forward along ``u = v^2 + v_x``. The price is locality: it is dense
where the Galerkin assembly is banded, and those two interior inverse mass matrices are
precisely the nonlocality that carries the Jacobi identity.

## Testing traps

Four of these are worth knowing before writing a test against any of this:

1. **Never use ``\mathfrak{so}(3)`` as a positive control** for a Jacobi or
   structure-constant routine. It lies in the six-parameter family
   ``c_{ij}^k = \epsilon_{ijl} n^{lk}`` with ``n`` **symmetric** — Bianchi class A — every
   member of which satisfies the Jacobi identity, and the perturbations one reaches for stay
   inside it: rescaling a generator, or rescaling a single structure constant, both keep ``n``
   symmetric and diagonal. So it goes on passing after it looks broken.

   Note that this is *not* because antisymmetry forces Jacobi in dimension three. A general
   antisymmetric `c` has nine parameters against this family's six, and 200 of 200 random ones
   fail. The degeneracy is the family, not the dimension. Use [`se3`](@ref) as the positive
   control and [`random_antisymmetric_c`](@ref) in dimension five as the negative one; see
   [Discrete Lie-Poisson brackets](@ref).
2. **An antisymmetric matrix has even rank.** For an even number of degrees of freedom the
   corank is two, and the extra sawtooth kernel vector is a *spurious* Casimir. Use an odd
   number.
3. **A uniform mesh confirms some identities for the wrong reason** — its assemblies are
   circulant. Use `RandomMesh` for anything claimed to hold on any mesh.
4. **Testing on a single smooth mode hides the cubic obstruction.** Excite the full spectrum
   with random degrees of freedom.
