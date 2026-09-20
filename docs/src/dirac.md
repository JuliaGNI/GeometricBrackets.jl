```@meta
CurrentModule = GeometricBrackets
```

# Dirac reduction

A truncated Lie-Poisson bracket closes on its coarse block but is not a Lie algebra —
[Discrete Lie-Poisson brackets](@ref) ends there. Dirac reduction is the obvious next move:
constrain the fine block away and see whether what is left satisfies the Jacobi identity.

It does not, and the reason is a theorem rather than an accident. This page is the Dirac
companion note, `discrete-lie-poisson-dirac-brackets.tex`, and the machinery it describes is
`src/dirac.jl` and `src/hierarchical.jl`.

## Setup

Split the coordinates into a coarse block ``V_1`` (`keep`) and a constrained block ``V_2``
(`con`), and impose ``u_a = 0`` for ``a \in V_2``. When the constraint matrix ``C =
\mathbb{J}_{22}`` is invertible — the constraints are **second class** — the Dirac bracket

```math
\mathbb{J}^*_{ij} = \mathbb{J}_{ij} - \mathbb{J}_{ia} \, (C^{-1})_{ab} \, \mathbb{J}_{bj}
```

has the constraints as Casimirs, and its restriction to ``V_1`` is the Schur complement

```math
\hat{\mathbb{J}} = \mathbb{J}_{11} - \mathbb{J}_{12} \, \mathbb{J}_{22}^{-1} \, \mathbb{J}_{21} .
```

[`dirac_blocks`](@ref) is the splitting, [`schur_complement`](@ref) the reduction,
[`dirac_tensor`](@ref) and [`dirac_R`](@ref) the full ``\mathbb{J}^*`` and the projector it is
built from, and [`reduced_bracket`](@ref) returns ``\hat{\mathbb{J}}`` together with the
derivative tensor ``\partial\hat{\mathbb{J}}`` that [`jacobi_residual`](@ref) needs.
[`dschur`](@ref) differentiates the Schur complement in a single coordinate.

Everything is written once and generically in the element type. With `Rational{BigInt}` the
identities come out exactly; with `Float64` the same code runs the refinement studies. The
Python prototypes carried an exact copy *and* a floating-point twin of each of these routines,
and recorded the two drifting apart — keeping one is the fix.

!!! note "Sign convention of the Jacobiator"
    [`jacobiator`](@ref) uses the convention of [`jacobi_residual`](@ref), ``\sum_l
    (\mathbb{J}_{il}\partial_l\mathbb{J}_{jk} + \text{cyclic})``, which for antisymmetric
    ``\mathbb{J}`` is the negative of the form used in the Python prototypes. Every claim these
    tools serve is either "this vanishes" or "these two agree", so a global sign is invisible to
    all of them — but it is worth knowing when comparing intermediate numbers against the
    Python.

## The central result: reduction is faithful, never curative

The Jacobiator transports **tensorially** through the Dirac projector:

```math
[\hat{\mathbb{J}},\hat{\mathbb{J}}]^{ijk}
  = \sum_{lmn} P^i_{\;l} \, P^j_{\;m} \, P^k_{\;n} \, [\mathbb{J},\mathbb{J}]^{lmn} ,
\qquad P = \Pi_{V_1} R .
```

[`project_jacobiator`](@ref) is that contraction. The consequence is immediate and decides the
whole question: **Dirac reduction preserves the Jacobi identity and cannot create it.** If
``\mathbb{J}`` is Poisson so is ``\hat{\mathbb{J}}``; if it is not, the reduced bracket fails
too, unless ``[\mathbb{J},\mathbb{J}]`` happens to land in ``\ker P^{\otimes 3}`` — a subspace
of dimension ``\dim V_2``.

```@example dirac
using GeometricBrackets

w = witt_truncation(2)                          # |k| ≤ 2 coarse, 2 < |k| ≤ 4 constrained
u = zeros(Rational{BigInt}, size(w.C, 1))
for i in w.keep
    u[i] = (i % 5 + 1)//3                       # a rational point on the constraint surface
end

Ĵ, dĴ = reduced_bracket(w.C, u, w.keep, w.con)
(before = jacobi_residual(lie_poisson_matrix(w.C, u), lie_poisson_derivative(w.C)),
 after  = jacobi_residual(Ĵ, dĴ))
```

Both exact rationals, both nonzero, and the second is not the smaller for any reason of
principle — `verify_dirac_reduction.jl` reports a normalised reduced Jacobiator of ``0.2770`` at
``K = 2`` and ``0.3879`` at ``K = 3``.

Nor is the correction a small perturbation one could hope to control: measured on the
hierarchical basis, ``\max|\hat{\mathbb{J}} - \mathbb{J}_{11}| / \max|\mathbb{J}_{11}| =
0.751``, so *consistency* is perturbed at leading order too. And the reduced bracket is
**rational** in ``u`` rather than linear, so it is not a Lie-Poisson bracket at all.

!!! warning "K = 1 is degenerate and proves nothing"
    At ``K = 1`` the reduced Jacobiator *does* vanish, and it is tempting to read that as the
    construction working at small size. It is not. ``\dim V_1 = 3``, so ``\hat{\mathbb{J}}``
    has rank two and is decomposable, and the Jacobi identity collapses to Frobenius
    involutivity — the same degeneracy that limits the Gardner-like four-bracket. Worse,
    ``\{L_{-1}, L_0, L_1\} = \mathfrak{sl}(2)`` is already a subalgebra, so the coarse block
    needs no ``V_2`` at all. Every branch of the repair search below finds its "solutions" here
    and nowhere else.

## Two obstructions before any computation

**Parity and the singular hypersurface.** ``C = \mathbb{J}_{22}`` is antisymmetric, so it has
even rank: with an odd ``\dim V_2`` it is singular everywhere and the Dirac bracket is
undefined. Even when ``\dim V_2`` is even, ``\det C`` vanishes on a hypersurface the flow can
cross. [`maximal_second_class`](@ref) finds the largest subset of the constraints that is
second class at a given point, and second-class subsets grow in pairs for the same reason.

**The closure premise fails for continuous finite elements.** The construction needs
``[V_1, V_1] \subseteq V_1 \oplus V_2``, and a ``C^0`` Lagrange basis cannot supply it: the
bracket differentiates, ``\varphi'`` jumps across element interfaces, and the product leaves the
space. [`closes_on`](@ref) and [`leak`](@ref) measure this. [`is_ideal`](@ref) and
[`restrict_c`](@ref) are the companion checks — note that restricting the tensor to `keep` is
*not* the same as the reduction, and `verify_dirac_reduction.jl` §1 keeps them apart.

## Closure and second-classness exclude each other

The two things the construction needs pull in opposite directions. Closure wants ``[V_1,V_1]``
to stay inside ``V_1 \oplus V_2``, which is a statement that the fine block is *reachable*;
second-classness wants ``\mathbb{J}_{22}`` nondegenerate, which is a statement that the fine
block is *self-coupled*. Section 5 of the companion note shows these cannot both hold in the
cases of interest, and the four realisations are the evidence.

## The four realisations

| realisation | closes | second class | verdict |
|:--|:--|:--|:--|
| Fourier on ``S^1`` — the Burgers / Witt bracket | yes | see §4 | fails |
| Fourier on ``\mathbb{T}^2`` — vorticity and Vlasov | yes | | fails |
| global polynomials ([`poly_truncation`](@ref)) | yes | | fails |
| the broken hierarchical space | yes, **exactly** | | fails |

The fourth is the one built here, and it is worth the space because it is the only one that gets
the closure premise for structural rather than accidental reasons.

### The broken hierarchical space

Take ``V_1`` the continuous degree-``p`` Lagrange space, embedded in the **broken**
(discontinuous Galerkin) degree-``2p-1`` space, and ``V_2`` its ``L^2``-orthogonal complement
there. Then

```math
[V_1, V_1] \subseteq V_1 \oplus V_2 \qquad \textbf{exactly},
```

because a product of two degree-``p`` polynomials with one derivative has degree ``2p-1``. That
is the closure premise a continuous basis cannot supply, obtained by giving up continuity rather
than by truncating.

```@example dirac
h = assemble_dg_hierarchical(1, 4)              # p = 1 on 4 cells, exact over ℚ
(dim = size(h.C, 1), coarse = length(h.keep), fine = length(h.con),
 closes = closes_on(h.C, h.keep, h.con), leak = leak(h.C, h.keep, h.con))
```

[`assemble_dg_hierarchical`](@ref) returns `(; C, keep, con, M)` ready for the reduction.
``V_2`` is obtained as `kernel(Q₁ * M_DG)` — exactly, where the floating-point prototype needed
an SVD — and the transformed commutator tensor is contracted one index at a time rather than by
the prototype's ``O(N_{\mathrm{DG}}^6)`` sextuple sum.

**Everything here is exact over ``\mathbb{Q}``, and no computer algebra system is involved.**
The shifted Legendre polynomials ``\tilde{L}_k(\xi) = P_k(2\xi - 1)`` have *integer*
coefficients,

```math
\tilde{L}_k(\xi) = \sum_{j=0}^{k} (-1)^{k+j} \binom{k}{j} \binom{k+j}{j} \xi^j ,
```

and ``\int_0^1 \xi^n \, d\xi = 1/(n+1)``, so every integral in the assembly is rational
arithmetic on coefficient vectors. [`shifted_legendre`](@ref) and
[`lagrange_coefficients`](@ref) return those vectors — in **ascending** powers throughout — and
[`dg_local_algebra`](@ref) and [`coarse_in_dg`](@ref) the local mass and commutator tensors and
the embedding of the coarse space. The Python prototype reached for SymPy here, and routed its
Lagrange shape functions through `float64` and `limit_denominator`; neither is needed, and
dropping both is what keeps a symbolic dependency out of the package altogether.

## The search for a repair

``\ker P^{\otimes 3}`` is not empty, so the question is whether some *variant* of the
construction aims into it. `scripts/search_dirac_variants.jl` searches, in order of prize:

| branch | idea | outcome |
|:--|:--|:--|
| B1a | graded Lie algebra structures on ``V_1 \oplus V_2``, exact | flatly inconsistent from ``K = 2`` on; its one solution at ``K = 1`` has ``[V_2,V_2] = 0``, hence ``C = 0`` and no Dirac bracket |
| B1b | the ``V_1^3`` equations alone, exact — they are *linear* in ``[V_2,V_1]`` | consistent, so the obstruction is **not** in the coarse equations |
| B1c | ungraded, by Gauss-Newton onto the Lie-algebra variety | solutions at ``K = 1``, where ``V_1 = \mathfrak{sl}(2)`` is already a subalgebra; none at ``K = 2`` |
| B2 | deform the constraint surface by a linear shear ``\phi_a = u_a + \sum B_{ai}u_i`` | nothing beyond the ``K=1`` degeneracy |
| B3 | choose ``V_2`` differently — every even-sized subset of a larger shell, exhaustive and exact | nothing beyond the ``K=1`` degeneracy (46 candidates / 12 hits at ``K=1``, 834 / 0 at ``K=2``) |
| B4 | nested reduction | ruled out by the composition lemma |

The script is **exploratory and slow**, and is deliberately excluded from `scripts/run_all.jl`.

## What survives

Two statements do hold, and they are the reason the machinery is kept:

  - **The discrete Poisson family of [Discrete Lie-Poisson brackets](@ref) is closed under
    Dirac reduction.** Reducing a ``g_i \mathbb{K}_{ij} g_j`` bracket gives another
    one.
  - **Dirac reduction of Zeitlin's bracket** is Poisson, because Zeitlin's bracket is — which is
    the tensorial-transport theorem read in the favourable direction.

And independently of the Jacobi identity: **the coarse subspace is exactly invariant** under the
reduced dynamics, for any Hamiltonian. That is what a Dirac bracket is for, and it is delivered.

## Exactness, cross-checked

`verify_dirac_reduction.jl`'s numbers ride on a random surface point, so a comparison against
the retired Python only establishes matching verdicts. Agreement was therefore established a
second way, at a **fixed rational point with no RNG on either side**: the reduced Jacobiator
``16134943910670/117497863``, ``\det C = 10490880625/50176``, the structure-constant residual
``10368/49``, and the broken-hierarchical mass entry ``M_{11} = 4/15`` all matched digit for
digit. That last one pins the exact shifted-Legendre and Lagrange assembly against the SymPy
version it replaces, and it is now a regression test in `test/dirac_tests.jl`. See
[Verification](@ref).

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["dirac.jl"]
```

### The broken hierarchical space

```@autodocs
Modules = [GeometricBrackets]
Pages = ["hierarchical.jl"]
```
