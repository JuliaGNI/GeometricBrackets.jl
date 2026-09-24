
@doc raw"""
    DiscreteBracket{T}

A discrete Poisson bracket, i.e. a matrix-valued function ``\mathbb{P}(\hat{u})`` through
which the bracket of two functions of the degrees of freedom is

```math
\{A, B\}_d = \sum_{i,j} \frac{\partial A}{\partial \hat{u}_i} \,
             \mathbb{P}_{ij} (\hat{u}) \, \frac{\partial B}{\partial \hat{u}_j} .
```

The interface is [`poisson_matrix`](@ref), [`poisson_apply`](@ref) and
[`poisson_derivative`](@ref); everything else — antisymmetry, the Jacobiator, the rank and
the Casimirs — follows from those and is written once.

# What is and is not guaranteed

Antisymmetry is a property of how the bracket is *assembled*, and every bracket here is
built in explicitly skew-symmetrised form, so it holds to the last bit for any quadrature
and any mesh. The Jacobi identity is not: it is a genuine question about the discretisation,
and the answer differs from one bracket to the next.

| bracket | antisymmetric | Jacobi |
|:--|:--|:--|
| [`ConstantBracket`](@ref) | yes | yes, automatically — a constant antisymmetric bracket satisfies it |
| [`GaugedBracket`](@ref) | yes | yes, by the theorem it is built on |
| [`MiuraBracket`](@ref) | yes | yes, being a pushforward of a Poisson bracket |
| [`AffineBracket`](@ref) | yes | **no** — and this is a result, not a defect |
| [`Arakawa`](@ref) | yes | **no** — the residual is of order one under refinement |

The failure of [`AffineBracket`](@ref) is the point of the KdV manuscript: the second
Hamiltonian structure of KdV is the Virasoro algebra, so the Jacobi identity would require
the discrete coefficients to close into a Lie algebra, a quadratic condition on the basis
that no choice of quadrature can deliver. The residual is of order one and flat under
refinement.
"""
abstract type DiscreteBracket{T} end

Base.eltype(::DiscreteBracket{T}) where {T} = T

"""
    poisson_matrix(bracket, û)

The dense structure matrix ``\\mathbb{P}(\\hat{u})``.
"""
function poisson_matrix end

"""
    poisson_apply(bracket, û, c)

The product ``\\mathbb{P}(\\hat{u}) \\, c``, formed without assembling the matrix where the
bracket admits it.
"""
function poisson_apply(b::DiscreteBracket, û::AbstractVector, c::AbstractVector)
    poisson_matrix(b, û) * c
end

@doc raw"""
    poisson_derivative(bracket, û)

The derivative tensor `dP[l,i,j]` ``= \partial \mathbb{P}_{ij} / \partial \hat{u}_l``.

This is what the Jacobiator needs, and it costs ``O(N^3)`` to store — which is why it is a
separate entry point rather than something [`poisson_matrix`](@ref) returns alongside. Use
it at the small `N` a structural test runs at, not in a time loop.
"""
function poisson_derivative end

"""
    isantisymmetric(bracket, û; atol = 1e-12)

Whether ``\\mathbb{P}(\\hat{u})`` is antisymmetric to within `atol`.
"""
function isantisymmetric(b::DiscreteBracket, û::AbstractVector; atol = 1e-12)
    P = poisson_matrix(b, û)
    maximum(abs, P + P') ≤ atol * max(one(eltype(P)), maximum(abs, P))
end

@doc raw"""
    jacobi_residual(bracket, û)

The normalised Jacobiator residual of ``\mathbb{P}`` at ``\hat{u}``,

```math
\max_{ijk} \left| \sum_l \left(
      \mathbb{P}_{il} \, \partial_l \mathbb{P}_{jk}
    + \mathbb{P}_{jl} \, \partial_l \mathbb{P}_{ki}
    + \mathbb{P}_{kl} \, \partial_l \mathbb{P}_{ij} \right) \right|
```

divided by the largest single term entering it, so that the number is dimensionless and
comparable across resolutions.

The normalisation is what makes the quantity worth reporting. A raw residual can be driven
to zero by any change that merely shrinks the bracket, and a residual normalised by
``\max|\mathbb{P}|^2`` by any change that makes it blow up; dividing by the largest term
actually formed is neither.

Returns `(residual, scale)` when `normalised = false`, so that both can be inspected.
"""
jacobi_residual(b::DiscreteBracket, û::AbstractVector; normalised::Bool = true) = jacobi_residual(
    poisson_matrix(b, û), poisson_derivative(b, û); normalised)

@doc raw"""
    jacobi_residual(P, dP; normalised = true)

The same quantity for a structure matrix and its derivative tensor `dP[l,i,j]`
``= \partial \mathbb{P}_{ij} / \partial \hat{u}_l`` given directly, without a
[`DiscreteBracket`](@ref) to carry them.

This is the entry point the structural scripts use, where the bracket is a Lie-Poisson
tensor built from structure constants rather than from a discretisation. It is generic in
the element type, so `Rational{BigInt}` gives an exactly zero residual where one is claimed
rather than a small floating-point number that has to be argued about.
"""
function jacobi_residual(P::AbstractMatrix, dP::AbstractArray{S, 3}; normalised::Bool = true) where {S}
    N = size(P, 1)
    T = promote_type(eltype(P), S)

    # A[i,j,k] = Σ_l P[i,l] ∂_l P[j,k]; the Jacobiator is the cyclic sum of A over (i,j,k)
    A = zeros(T, N, N, N)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(T)
        for l in 1:N
            s += P[i, l] * dP[l, j, k]
        end
        A[i, j, k] = s
    end

    res = zero(real(T))
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        res = max(res, abs(A[i, j, k] + A[j, k, i] + A[k, i, j]))
    end

    scale = maximum(abs, A)
    normalised || return (res, scale)
    iszero(scale) ? zero(res) : res / scale
end

@doc raw"""
    structure_constant_residual(C)

The normalised violation of the structure-constant condition by the three-tensor
`C[m,i,j]` ``= c_{ij}^m``,

```math
\max \left| \sum_m \left( c_{ij}^m c_{mk}^n
                        + c_{jk}^m c_{mi}^n
                        + c_{ki}^m c_{mj}^n \right) \right| ,
```

normalised by the largest term formed.

A bracket linear in the field, ``\mathbb{J}_{ij}(u) = \sum_m c_{ij}^m u_m``, is Poisson if
and only if the ``c_{ij}^m`` are the structure constants of a Lie algebra, which is exactly
this condition.

# Choosing a control

Do **not** validate a routine like this against ``\mathfrak{so}(3)``. It lies inside the
six-parameter family ``c_{ij}^k = \epsilon_{ijl} n^{lk}`` with ``n`` symmetric — Bianchi
class A — *every* member of which satisfies the Jacobi identity, and the perturbations one
naturally reaches for stay inside it: rescaling a generator, or rescaling a single structure
constant, both leave ``n`` symmetric and diagonal. So ``\mathfrak{so}(3)`` keeps passing
after it has apparently been broken, and passing tells you nothing.

It is *not* the case that antisymmetry alone forces Jacobi in three dimensions — a general
antisymmetric `c` has nine parameters against this family's six, and random ones fail
comfortably. The degeneracy is specific to the family, not to the dimension. (The Python
prototypes state the broader claim; it is an erratum, recorded in `docs/src/verification.md`.)

Use ``\mathfrak{se}(3)`` as the positive control, and a random antisymmetric `c` in
dimension five as the negative one.
"""
function structure_constant_residual(C::AbstractArray{T, 3}; normalised::Bool = true) where {T}
    N = size(C, 1)
    # A[n,i,j,k] = Σ_m C[m,i,j] C[n,m,k]
    A = zeros(T, N, N, N, N)
    @inbounds for k in 1:N, j in 1:N, i in 1:N, n in 1:N
        s = zero(T)
        for m in 1:N
            s += C[m, i, j] * C[n, m, k]
        end
        A[n, i, j, k] = s
    end

    res = zero(real(T))
    @inbounds for k in 1:N, j in 1:N, i in 1:N, n in 1:N
        res = max(res, abs(A[n, i, j, k] + A[n, j, k, i] + A[n, k, i, j]))
    end

    scale = maximum(abs, A)
    normalised || return (res, scale)
    iszero(scale) ? zero(res) : res / scale
end

## Constant bracket

@doc raw"""
    ConstantBracket(P)
    ConstantBracket(space, K)

A bracket whose structure matrix does not depend on the field,
``\mathbb{P}(\hat{u}) = \mathbb{P}``.

The second form sandwiches a weak-form operator between two inverse mass matrices,
``\mathbb{P} = \mathbb{M}^{-1} \mathbb{K} \mathbb{M}^{-1}``, which is what the discrete
functional derivative produces, and skew-symmetrises `K` on the way in.

Such a bracket is **automatically Poisson**: the Jacobi identity for a constant bracket
follows from antisymmetry alone, every derivative in the Jacobiator being zero. This is why
the first KdV structure survives discretisation intact where the second does not.

```jldoctest
julia> s = SplineSpace(16, 3);

julia> P = kdv_bracket_1(s);

julia> isantisymmetric(P, zeros(16)), jacobi_residual(P, zeros(16)) == 0
(true, true)
```
"""
struct ConstantBracket{T, MT <: AbstractMatrix{T}} <: DiscreteBracket{T}
    P::MT

    function ConstantBracket(P::MT) where {T, MT <: AbstractMatrix{T}}
        size(P, 1) == size(P, 2) || throw(ArgumentError(
            "a structure matrix must be square, got $(size(P))"))
        new{T, MT}(P)
    end
end

function ConstantBracket(s::DiscreteSpace, K::AbstractMatrix)
    Minv = inverse_mass_matrix(s)
    ConstantBracket(Minv * _skew(K) * Minv)
end

"""
    _skew(A)

The antisymmetric part ``\\tfrac{1}{2}(A - A^T)``.

Written out rather than skipped where `A` happens already to be antisymmetric: the factor of
one half belongs to the definition of the skew-symmetrised bracket, and dropping it because
`A - Aᵀ` equals `2A` on a periodic mesh would make the code right for a reason that does not
survive a change of basis.
"""
_skew(A::AbstractMatrix) = (A - A') / 2

poisson_matrix(b::ConstantBracket, û::AbstractVector) = b.P
poisson_matrix(b::ConstantBracket) = b.P
poisson_apply(b::ConstantBracket, û::AbstractVector, c::AbstractVector) = b.P * c

function poisson_derivative(b::ConstantBracket{T}, û::AbstractVector) where {T}
    zeros(T, size(b.P, 1), size(b.P)...)
end

function jacobi_residual(b::ConstantBracket{T}, û::AbstractVector; normalised::Bool = true) where {T}
    normalised ? zero(T) : (zero(T), zero(T))
end

Base.size(b::ConstantBracket) = size(b.P)

## Affine bracket

@doc raw"""
    AffineBracket(space, scale, Ψ, K0)

A bracket whose structure matrix is affine in the field,

```math
\mathbb{P}(\hat{u}) = \mathbb{M}^{-1}
    \left( \mathbb{K}_0 + \mathbb{K}^\rho (\hat{u}) \right) \mathbb{M}^{-1} ,
```

with `K0` a constant antisymmetric block and

```math
\mathbb{K}^\rho_{kl} (\hat{u}) = \sigma \int_\Omega \rho_h
    \left( \phi_k \partial_x \phi_l - \phi_l \partial_x \phi_k \right) dx
```

the part of transport type, ``\sigma (\rho \partial_x + \partial_x \rho)`` in weak form.
Both blocks are skew-symmetrised on the way in, so the assembled matrix is antisymmetric
identically.

The density is given by its own tabulation `Ψ`, with ``\rho_h = \Psi^T \hat{u}`` at the
quadrature points, which keeps the bracket *linear* in ``\hat{u}`` and so lets both the
matrix-free product and the Jacobian of the flow be written in closed form:

  - for the second KdV bracket, ``4u \partial_x + 2u_x = 2(u \partial_x + \partial_x u)``,
    so `scale` is `2` and ``\Psi = \Phi_0``, the density being ``u_h`` itself;
  - for the second Camassa-Holm bracket, ``m \partial_x + \partial_x m``, `scale` is `1`
    and ``\Psi = \Phi_0 - \Phi_2``, the density being ``m_h = u_h - u_h''``.

# Why the skew-symmetrised form

The unsymmetrised assembly ``\int (4 u_h \phi_k \partial_x \phi_l + 2 \partial_x u_h \phi_k
\phi_l)`` is the same matrix in exact arithmetic — the term ``2 \partial_x u_h \phi_k
\phi_l`` is symmetric in ``k \leftrightarrow l`` and drops out of the skew part, while
integrating it by parts supplies the rest. But it is antisymmetric only if the quadrature
integrates the total derivative ``\partial_x (u_h \phi_k \phi_l)``, of degree ``3p-1``,
exactly. Written as above, antisymmetry holds identically at any `nq` and on any mesh, and
the exact conservation of the generating Hamiltonian holds with it. That is the whole
practical content of the skew-symmetrisation, and it is not a cosmetic rearrangement.

Neither of these brackets satisfies the Jacobi identity; see [`DiscreteBracket`](@ref).
"""
struct AffineBracket{
    T, ST <: DiscreteSpace{T}, PT <: AbstractMatrix{T}, KT <: AbstractMatrix{T}} <:
       DiscreteBracket{T}
    space::ST
    scale::T
    Ψ::PT
    K0::KT
    Minv::Matrix{T}

    # `Ψ` and `K0` are both stored AS GIVEN rather than densified. `Ψ` is the basis
    # tabulation, which both spaces hand over sparse -- under one per cent full at the
    # resolutions these runs reach -- and it is contracted on every Newton iteration by
    # `kernel_matrix` and `kernel_directional`. Calling `Matrix` on it here, as this
    # constructor used to, threw that away and made each of those contractions cost N² per
    # quadrature point instead of (p+1)².
    #
    # `K0` is the same story, and it used to be densified for the same bad reason. It is
    # `mixed_matrix(s, 1, 2)` for KdV, banded with circular bandwidth p, and zero for
    # Camassa-Holm; `_skew` preserves that. Keeping it sparse is what makes
    # `kernel_operator` -- and hence the whole mixed formulation, which never forms
    # `Minv K Minv` at all -- banded rather than dense. It also makes `poisson_apply`'s
    # `K0 * v` an O(Np) matvec instead of O(N²), which the explicit runs pay four times a
    # step for millions of steps.
    #
    # `Minv` stays dense, because `M⁻¹` of a banded matrix genuinely is full. It is needed
    # only by the analysis paths (`poisson_matrix`, `poisson_tensor`, `kernel_tensor`); the
    # dynamics can always solve with `M` instead. See `kernel_operator`.
    function AffineBracket(s::ST, scale::Real, Ψ::AbstractMatrix,
            K0::AbstractMatrix) where {T, ST <: DiscreteSpace{T}}
        size(Ψ) == size(basis_values(s, 0)) || throw(DimensionMismatch(
            "the density table has size $(size(Ψ)) but the basis tabulation has " *
            "$(size(basis_values(s, 0)))"))
        K̄ = _skew(K0)
        new{T, ST, typeof(Ψ), typeof(K̄)}(s, convert(T, scale), Ψ, K̄,
            inverse_mass_matrix(s))
    end
end

"""
    density(b::AffineBracket, û)

The values of the density ``\\rho_h`` at the quadrature points.
"""
density(b::AffineBracket, û::AbstractVector) = b.Ψ' * û

"""
    kernel_matrix(b::AffineBracket, û)

The `u`-dependent block ``\\mathbb{K}^\\rho(\\hat{u})``, an antisymmetric `N × N` matrix.
"""
function kernel_matrix(b::AffineBracket, û::AbstractVector)
    A = weighted_matrix(b.space, density(b, û), 0, 1)
    b.scale * (A - A')
end

@doc raw"""
    kernel_apply(b::AffineBracket, û, c)

The `u`-dependent block applied to `c`, formed without assembling it.

With ``c_h`` the field of `c`, the contraction is
``\sigma \int \rho_h (\phi_k c_h' - c_h \phi_k')``, which costs ``O(NQ)`` against the
``O(N^2 Q)`` of assembling the block first. In an explicit run the field is evaluated four
times per step and the step counts reach the millions, and this is where that time goes.
"""
function kernel_apply(b::AffineBracket, û::AbstractVector, c::AbstractVector)
    s = b.space
    w = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    Φ1 = basis_values(s, 1)
    ρ = density(b, û)
    ch = Φ0' * c
    cx = Φ1' * c
    b.scale * (Φ0 * (w .* ρ .* cx) - Φ1 * (w .* ρ .* ch))
end

@doc raw"""
    kernel_tensor(b::AffineBracket)

The three-tensor `Ku[m,k,l]` of the `u`-linear block, so that the block at ``\hat{u}`` is
``\sum_m \hat{u}_m \, \mathrm{Ku}[m,\cdot,\cdot]``.

``O(N^3)`` in storage; only the Jacobiator needs it.
"""
function kernel_tensor(b::AffineBracket{T}) where {T}
    s = b.space
    w = quadrature_weights(s)
    # Densified deliberately, and only here. The tabulations are stored sparse because the
    # assemblies contract them, and that is the hot path; this routine instead RANDOM-ACCESSES
    # them N³ times, where every sparse read is a binary search down a column. It builds an
    # N³ tensor, so it is off the time loop by construction -- `bracket_directional` is what
    # a Jacobian uses -- and dense reads are what the loop below actually wants.
    Φ0 = Matrix(basis_values(s, 0))
    Φ1 = Matrix(basis_values(s, 1))
    Ψ = Matrix(b.Ψ)
    N = nbasis(s)

    T1 = zeros(T, N, N, N)
    @inbounds for l in 1:N, kk in 1:N, m in 1:N
        acc = zero(T)
        for r in eachindex(w)
            acc += w[r] * Ψ[m, r] * Φ0[kk, r] * Φ1[l, r]
        end
        T1[m, kk, l] = acc
    end
    Ku = similar(T1)
    @inbounds for l in 1:N, kk in 1:N, m in 1:N
        Ku[m, kk, l] = b.scale * (T1[m, kk, l] - T1[m, l, kk])
    end
    return Ku
end

@doc raw"""
    kernel_directional(b::AffineBracket, v)

The matrix ``D_{km} = \partial (\mathbb{K}^\rho (\hat{u}) v)_k / \partial \hat{u}_m`` at
fixed `v`, which is independent of ``\hat{u}`` because the block is linear in it,

```math
D_{km} = \sigma \int_\Omega \psi_m
    \left( \phi_k \partial_x v_h - v_h \partial_x \phi_k \right) dx .
```

This is what makes the Jacobian of the flow analytic and ``O(N^2 Q)`` rather than a
contraction of the ``O(N^3)`` tensor.
"""
function kernel_directional(b::AffineBracket, v::AbstractVector)
    s = b.space
    w = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    Φ1 = basis_values(s, 1)
    vh = Φ0' * v
    vx = Φ1' * v
    b.scale * (Φ0 * Diagonal(w .* vx) * b.Ψ' - Φ1 * Diagonal(w .* vh) * b.Ψ')
end

@doc raw"""
    kernel_operator(b::AffineBracket, û)

The weak-form block ``\mathbb{K}(\hat{u}) = \mathbb{K}_0 + \mathbb{K}^\rho(\hat{u})``, without
the surrounding inverse mass matrices.

This is [`poisson_matrix`](@ref)'s middle factor, and unlike `poisson_matrix` it is **sparse**:
both terms are banded with circular bandwidth `p`, so at ``N = 384`` and ``p = 3`` it is under
two per cent full. The densification is entirely in the sandwich
``\mathbb{M}^{-1} \mathbb{K} \mathbb{M}^{-1}``, which is what the mixed formulation exists to
avoid — see [`Integrator`](@ref)'s `formulation` keyword.

Together with [`kernel_directional`](@ref), which is ``\partial(\mathbb{K}^\rho v)/\partial\hat{u}``
and likewise banded, this is everything the mixed Jacobian needs from the bracket.
"""
kernel_operator(b::AffineBracket, û::AbstractVector) = b.K0 + kernel_matrix(b, û)

"""
    kernel_operator(b::DiscreteBracket, û)

Throws: there is no weak-form block to return for the brackets other than
[`AffineBracket`](@ref).

[`ConstantBracket`](@ref) and [`GaugedBracket`](@ref) both store the *sandwiched*
``\\mathbb{M}^{-1}\\mathbb{K}\\mathbb{M}^{-1}``, formed once in the constructor, and
[`MiuraBracket`](@ref) stores a congruence ``\\mathbb{L}\\mathbb{P}_1\\mathbb{L}^T`` of one, so
``\\mathbb{K}`` cannot be recovered from any of them without inverting the mass matrix again.
Only [`AffineBracket`](@ref) — the KdV and Camassa-Holm second brackets — supports the mixed
formulation, and this method is what makes asking for it anywhere else an `ArgumentError`
rather than a silently different method.
"""
function kernel_operator(b::DiscreteBracket, ::AbstractVector)
    throw(ArgumentError(
        "kernel_operator is only defined for an AffineBracket; $(nameof(typeof(b))) stores the " *
        "sandwiched Minv*K*Minv rather than K, so the mixed formulation is not available for it"))
end

function poisson_matrix(b::AffineBracket, û::AbstractVector)
    b.Minv * (b.K0 + kernel_matrix(b, û)) * b.Minv
end

function poisson_apply(b::AffineBracket, û::AbstractVector, c::AbstractVector)
    (v = b.Minv * c; b.Minv * (b.K0 * v + kernel_apply(b, û, v)))
end

@doc raw"""
    poisson_tensor(b::AffineBracket)

The pair `(P0, C)` with ``\mathbb{P}(\hat{u}) = \mathbb{P}_0 + \sum_m \hat{u}_m C[m]``,
``\mathbb{P}_0 = \mathbb{M}^{-1} \mathbb{K}_0 \mathbb{M}^{-1}`` and
``C[m] = \mathbb{M}^{-1} \mathrm{Ku}[m] \mathbb{M}^{-1}``.
"""
function poisson_tensor(b::AffineBracket{T}) where {T}
    Ku = kernel_tensor(b)
    N = nbasis(b.space)
    C = zeros(T, N, N, N)
    @inbounds for m in 1:N
        C[m, :, :] = b.Minv * Ku[m, :, :] * b.Minv
    end
    (b.Minv * b.K0 * b.Minv, C)
end

poisson_derivative(b::AffineBracket, û::AbstractVector) = poisson_tensor(b)[2]

Base.size(b::AffineBracket) = (nbasis(b.space), nbasis(b.space))

## Gauged bracket

@doc raw"""
    GaugedBracket(K, g, dg)

The bracket of the discretisation theorem for Lie-Poisson brackets,

```math
\mathbb{J}_{ij} (u) = g_i (u_i) \, \mathbb{K}_{ij} \, g_j (u_j)
\qquad \text{(no sum over \$i\$ and \$j\$)} ,
```

with ``\mathbb{K}`` a **constant** antisymmetric matrix and ``g`` a smooth function of the
**single** variable ``u_i``.

Such a bracket is Poisson on any domain where `g` is defined. Antisymmetry is immediate, and
the Jacobi identity holds because the bracket is the pullback of the constant bracket
``\mathbb{K}`` under a change of variables that acts coordinate by coordinate — a
transformation of a Poisson bracket is again a Poisson bracket.

# Both hypotheses are sharp

Neither can be relaxed. If ``\mathbb{K}`` depends on ``u``, or if ``g_i`` depends on any
variable other than ``u_i``, the Jacobi identity fails; the test suite exhibits both
failures rather than merely asserting the theorem.

The Burgers bracket is the case ``g(u) = \sqrt{u}``, which requires ``u > 0``. Its Casimirs
are ``C_n = \sum_i n_i G_i(u_i)`` with ``G' = 1/g`` and ``n`` in the kernel of
``\mathbb{K}``; for the square root, ``G(u) = 2\sqrt{u}``.

`dg` is ``g'``, needed for the Jacobiator and for the Jacobian of the flow.
"""
struct GaugedBracket{T, GT, DT} <: DiscreteBracket{T}
    K::Matrix{T}
    g::GT
    dg::DT

    function GaugedBracket(K::AbstractMatrix{T}, g::GT, dg::DT) where {T, GT, DT}
        size(K, 1) == size(K, 2) || throw(ArgumentError(
            "a structure matrix must be square, got $(size(K))"))
        new{T, GT, DT}(Matrix(_skew(K)), g, dg)
    end
end

function poisson_matrix(b::GaugedBracket, û::AbstractVector)
    (gv = b.g.(û); Diagonal(gv) * b.K * Diagonal(gv))
end

function poisson_apply(b::GaugedBracket, û::AbstractVector, c::AbstractVector)
    (gv = b.g.(û); gv .* (b.K * (gv .* c)))
end

function poisson_derivative(b::GaugedBracket{T}, û::AbstractVector) where {T}
    N = length(û)
    gv = b.g.(û)
    dv = b.dg.(û)
    dP = zeros(T, N, N, N)
    # ∂J_ij/∂u_l = δ_il g'(u_i) K_ij g_j + δ_jl g_i K_ij g'(u_j)
    @inbounds for j in 1:N, i in 1:N

        dP[i, i, j] += dv[i] * b.K[i, j] * gv[j]
        dP[j, i, j] += gv[i] * b.K[i, j] * dv[j]
    end
    return dP
end

Base.size(b::GaugedBracket) = size(b.K)

## Miura bracket

@doc raw"""
    MiuraBracket(space, v̂, bracket1)

The pushforward of the first bracket along the Miura map, evaluated at the mKdV field `v̂`,

```math
\mathbb{P}^2_{\mathrm{M}} (\hat{v}) = \mathbb{L}(\hat{v}) \; \mathbb{P}^1 \;
                                      \mathbb{L}(\hat{v})^T ,
\qquad
\mathbb{L}(\hat{v}) = -\mathbb{M}^{-1} \big( 2\mathbb{B}(\hat{v}) + S \big) ,
\qquad
\mathbb{B}_{kl} = \int_\Omega v_h \phi_k \phi_l \, dx ,
```

the discrete counterpart of the Miura factorisation

```math
(2v + \partial_x) \, \partial_x \, (2v - \partial_x) = -4u\partial_x - 2u_x - \partial_x^3
```

under ``u = -(v^2 + v_x)``. The overall sign of ``\mathbb{L}`` cancels in
``\mathbb{L}\mathbb{P}^1\mathbb{L}^T``, so the assembled bracket is what it always was; it
is the *chart* that carries the sign, and `miura_invert` and the chain rule below need the
true Jacobian.

Antisymmetry is a one-line consequence of ``\mathbb{B}^T = \mathbb{B}`` and ``S^T = -S``,
and the Jacobi identity of the facts that ``\mathbb{P}^1`` is constant and antisymmetric and
``\mathbb{L}`` invertible. The normalised Jacobiator is at round-off at every resolution,
against ``\approx 0.42`` flat for the Galerkin [`kdv_bracket_2`](@ref).

# The price is locality

Both brackets have the form ``\mathbb{M}^{-1}\mathbb{K}\mathbb{M}^{-1}``, but the Galerkin
``\mathbb{K}`` is a single local assembly, banded with the bandwidth of the basis, whereas

```math
\mathbb{K}_{\mathrm{M}} = \big( 2\mathbb{B} + S \big) \, \mathbb{M}^{-1} S \, \mathbb{M}^{-1}
                          \, \big( 2\mathbb{B} - S \big)
```

carries **two interior inverse mass matrices** and is dense: for `N = 24` cubic splines the
circular bandwidth is 3 for the Galerkin assembly and 12, the maximum, for
``\mathbb{K}_{\mathrm{M}}``. That nonlocality is not an artefact to be optimised away — it is
what the Jacobi identity costs.

# It is a bracket on v-space

``\mathbb{P}^2_{\mathrm{M}}`` is a function of ``\hat{v}``, so writing it as a structure on
``u``-space presupposes ``\mathcal{M}_h^{-1}``, which is undefined on most of
``\mathbb{R}^N``; see [`miura_invert`](@ref) and [`hill_lambda0`](@ref). In practice one does
not form this matrix at all but integrates [`MiuraSystem`](@ref) in the ``\hat{v}`` chart,
where the bracket is the constant ``\mathbb{P}^1``.

See also [`miura_derivative`](@ref) for ``\mathbb{L}`` and [`miura_map`](@ref) for
``\mathcal{M}_h``.
"""
struct MiuraBracket{T, ST <: DiscreteSpace{T}} <: DiscreteBracket{T}
    space::ST
    v̂::Vector{T}
    L::Matrix{T}
    P1::Matrix{T}
    P::Matrix{T}

    function MiuraBracket(s::ST, v̂::AbstractVector, b1::ConstantBracket{T}) where
            {T, ST <: DiscreteSpace{T}}
        L = miura_derivative(s, v̂)
        P1 = Matrix(poisson_matrix(b1))
        P = L * P1 * L'
        new{T, ST}(s, collect(T, v̂), L, P1, Matrix(_skew(P)))
    end
end

MiuraBracket(s::DiscreteSpace, v̂::AbstractVector) = MiuraBracket(s, v̂, kdv_bracket_1(s))

poisson_matrix(b::MiuraBracket, û::AbstractVector) = b.P
poisson_matrix(b::MiuraBracket) = b.P
poisson_apply(b::MiuraBracket, û::AbstractVector, c::AbstractVector) = b.P * c

@doc raw"""
    poisson_derivative(b::MiuraBracket, û)

The derivative tensor of ``\mathbb{P}^2_{\mathrm{M}}`` **with respect to ``\hat{u}``**.

This is the derivative that the Jacobiator needs, and computing it is the whole content of
the claim that the Miura bracket is Poisson. The matrix is a function of ``\hat{v}``, so the
chain rule goes through the chart,

```math
\frac{\partial \mathbb{P}^2_{\mathrm{M}}}{\partial \hat{u}_m}
  = \sum_l \frac{\partial \mathbb{P}^2_{\mathrm{M}}}{\partial \hat{v}_l}
           \left( \mathbb{L}^{-1} \right)_{lm} ,
\qquad
\frac{\partial \mathbb{P}^2_{\mathrm{M}}}{\partial \hat{v}_l}
  = \frac{\partial \mathbb{L}}{\partial \hat{v}_l} \mathbb{P}^1 \mathbb{L}^T
  + \mathbb{L} \mathbb{P}^1 \frac{\partial \mathbb{L}^T}{\partial \hat{v}_l} ,
```

with ``\partial \mathbb{L} / \partial \hat{v}_l = 2 \mathbb{M}^{-1} \mathbb{T}^l`` and
``\mathbb{T}^l_{km} = \int_\Omega \phi_l \phi_k \phi_m \, dx``, since ``\mathbb{B}`` is
affine in ``\hat{v}``.

Returning zero here — as a bracket that happened to be constant in ``\hat{u}`` would — would
make [`jacobi_residual`](@ref) report success for a tautological reason. It is computed.
"""
function poisson_derivative(b::MiuraBracket{T}, û::AbstractVector) where {T}
    s = b.space
    N = nbasis(s)
    w = quadrature_weights(s)
    # dense for the same reason as in `kernel_tensor`: the triple-product tensor below reads
    # the tabulation N³ times at random rather than contracting it
    Φ0 = Matrix(basis_values(s, 0))
    Minv = inverse_mass_matrix(s)
    Linv = inv(b.L)

    # the triple-product tensor T[l,k,m] = ∫ φ_l φ_k φ_m
    Tt = zeros(T, N, N, N)
    @inbounds for m in 1:N, k in 1:N, l in 1:N
        acc = zero(T)
        for r in eachindex(w)
            acc += w[r] * Φ0[l, r] * Φ0[k, r] * Φ0[m, r]
        end
        Tt[l, k, m] = acc
    end

    # ∂P/∂v̂_l, one N × N matrix per l
    dPdv = Vector{Matrix{T}}(undef, N)
    @inbounds for l in 1:N
        # ∂L/∂v̂_l = -2 M⁻¹ Tˡ, the sign being that of `miura_derivative`
        dL = -2 .* (Minv * Tt[l, :, :])
        A = dL * b.P1 * b.L'
        # the second term is L P¹ dLᵀ = -(dL P¹ Lᵀ)ᵀ, because P¹ is antisymmetric, so the
        # two SUBTRACT; adding them would give a symmetric matrix and a Jacobiator of
        # order one for a bracket that is in fact Poisson
        dPdv[l] = A - A'
    end

    # chain through the chart
    dP = zeros(T, N, N, N)
    @inbounds for m in 1:N
        acc = zeros(T, N, N)
        for l in 1:N
            acc .+= Linv[l, m] .* dPdv[l]
        end
        dP[m, :, :] = acc
    end
    return dP
end

Base.size(b::MiuraBracket) = size(b.P)

"""
    miura_moment_matrix(space, v̂)

The symmetric matrix ``\\mathbb{B}_{kl} = \\int_\\Omega v_h \\phi_k \\phi_l \\, dx``.
"""
function miura_moment_matrix(s::DiscreteSpace, v̂::AbstractVector)
    weighted_matrix(s, field(s, v̂), 0, 0)
end

@doc raw"""
    miura_derivative(space, v̂)

The Jacobian of the discrete Miura map,
``\mathbb{L}(\hat{v}) = \mathbb{M}^{-1}(2\mathbb{B}(\hat{v}) + S)``, the discretisation of
``\mathcal{M}' = 2v + \partial_x``.

Its transpose ``\mathbb{L}^T = (2\mathbb{B} - S)\mathbb{M}^{-1}`` discretises the adjoint
``2v - \partial_x``.

# Where it degenerates, and how sharply that is known

For the *differential* operator the statement is exact: ``\ker(2v + \partial_x)`` is spanned
by ``\exp(-2\int v)``, which is periodic precisely on ``\int_\Omega v \, dx = 0``. For the
assembled ``\mathbb{M}^{-1}(2\mathbb{B} + S)`` it is **evidence rather than proof** — the
condition number is around ``10^9`` on that set and below ``10^2`` away from it — and the
discrete critical set need not coincide with the continuous one.

What *is* exact here is a statement in the other direction, and it is the one that matters in
practice: ``\int_\Omega v_h \, dx`` is an exact Casimir of ``\mathbb{P}^1`` in this chart
(see [`miura_casimir`](@ref)), so the Miura flow **cannot reach** ``\int_\Omega v_h = 0``
from initial data off it, at any resolution and for any step lying in the range of
``\mathbb{P}^1``. The near-degeneracy is therefore not something a run can wander into; it
has to be posed there.
"""
function miura_derivative(s::DiscreteSpace, v̂::AbstractVector)
    Minv = inverse_mass_matrix(s)
    .-(Minv * (2 .* miura_moment_matrix(s, v̂) .+ derivative_matrix(s)))
end

@doc raw"""
    miura_map(space, v̂, λ = 0)

The discrete Miura map with spectral parameter,
``\hat{u} = \mathcal{M}_h^\lambda(\hat{v})``, i.e. the ``L^2`` projection of
``-(v_h^2 + v_{h,x}) - \lambda``. The sign is the one that makes the map carry the mKdV flow
``v_t = 6v^2v_x - v_{xxx}`` to ``u_t + 6uu_x + u_{xxx} = 0``.

# What λ is for

At ``\lambda = 0`` the image is the half-space ``C_{0,d} \le 0`` and, sharply, the fields
whose Hill operator is positive definite — which excludes every standard KdV example. The
parameter removes that restriction, and it is not an ad-hoc shift: the Riccati substitution
``v = \psi_x/\psi`` turns ``u = -(v^2 + v_x) - \lambda`` into

```math
\left( -\partial_x^2 - u_h \right) \psi = \lambda \psi ,
```

so **λ is an eigenvalue parameter of the very Hill operator that decides invertibility**. A
real periodic preimage exists exactly when ``\lambda`` lies at or below the bottom of the
spectrum,

```math
\lambda < \lambda_0 \left( -\partial_x^2 - u_h \right) ,
```

which for any ``\hat{u}`` whatever can be arranged by taking ``\lambda`` negative enough.
[`miura_lambda`](@ref) returns such a value. At ``\lambda = 0`` the condition reduces to
``\lambda_0 > 0``, the criterion of [`hill_lambda0`](@ref).

# What it costs

``\mathbb{L}`` does not see a constant, so [`miura_derivative`](@ref) and hence
[`MiuraBracket`](@ref) are **unchanged**. What changes is which bracket the pushforward is:

```math
\mathbb{L} \mathbb{P}^1 \mathbb{L}^T = \mathbb{P}^2 - 4\lambda \mathbb{P}^1 ,
```

a member of the bi-Hamiltonian *pencil* rather than ``\mathbb{P}^2`` itself. It is exactly
Poisson for every ``\lambda``, by the same one-line argument. On the ``u`` side the extra
``-4\lambda\mathbb{P}^1 \partial H_2/\partial\hat{u}`` is a constant-speed translation, so
the trajectory is the KdV solution in a uniformly moving frame and all three invariants are
untouched — ``H_2`` and ``C_{0}`` because the drift is generated by them, ``H_1`` because
``\{H_1,H_2\}_1 = 0``.

# The mass identity, and the half-space it used to confine the image to

Pairing with the partition of unity gives the exact identity

```math
C_{0,d} = \int_\Omega u_h \, dx
        = -\int_\Omega \big( v_h^2 + \partial_x v_h \big) \, dx - \lambda L
        = -\int_\Omega v_h^2 \, dx - \lambda L ,
```

so at ``\lambda = 0`` the image lies in the half-space of **non-positive mass**,
``C_{0,d} \le 0``, meeting ``C_{0,d} = 0`` only at ``v_h \equiv 0``. A negative ``\lambda``
moves that half-space up by ``-\lambda L`` and is what lets any mass be reached.

!!! note "The sign convention never decided this"
    In the older ``u_t = 6uu_x - u_{xxx}`` convention the map was ``u = v^2 + v_x`` and the
    image was the *non-negative* half-space, with the solitons excluded for being
    depressions. Flipping to elevations flips the half-space with them, and they are excluded
    again — now for being positive. The obstruction was never a matter of signs: a soliton is
    by construction a reflectionless potential *with* a bound state, and the ``\lambda = 0``
    chart covers exactly the fields whose Hill operator has none. What lifts the restriction
    is ``\lambda``, because it moves the spectral threshold rather than relabelling it.
"""
miura_map(s::DiscreteSpace, v̂::AbstractVector, λ::Real = 0) = project(
    s, .-(field(s, v̂) .^ 2 .+ field(s, v̂, 1))) .- λ

@doc raw"""
    hill_lambda0(space, û)

The lowest eigenvalue of the Galerkin Hill operator ``-\partial_x^2 - u_h``, from
``(\mathbb{K}^1 - \mathbb{U})\psi = \lambda \mathbb{M}\psi`` with
``\mathbb{U}_{kl} = \int u_h \phi_k \phi_l``.

This is the **sharp** criterion for [`miura_invert`](@ref) to succeed: the Riccati
substitution ``v = \psi_x/\psi`` turns ``u = -(v^2 + v_x)`` into ``\psi_{xx} = -u\psi``, and
a nodeless solution — hence one with a real, periodic logarithmic derivative — exists exactly
when zero lies below the spectrum, i.e. when this eigenvalue is positive. The potential is
``-u_h`` rather than ``u_h`` because the convention here is
``u_t + 6uu_x + u_{xxx} = 0``.

```jldoctest
julia> s = SplineSpace(20, 3);

julia> λ(c) = hill_lambda0(s, project(s, x -> c + sin(x) + 0.4cos(2x)));

julia> λ(-0.6) > 0 && λ(-0.3) < 0    # the threshold lies between
true
```
"""
function hill_lambda0(s::DiscreteSpace, û::AbstractVector)
    K1 = stiffness_matrix(s)
    # the Riccati substitution turns u = -(v² + v_x) into ψ_xx = -u ψ, so the Hill potential
    # is MINUS the KdV field in this convention
    U = .-weighted_matrix(s, field(s, û), 0, 0)
    # densified on the way in: the assemblies are sparse, and a sparse Cholesky cannot
    # solve against a sparse right-hand side. The eigenvalue problem is dense in any case.
    minimum(real.(eigvals(mass_factorization(s) \ Matrix(K1 .+ U))))
end

@doc raw"""
    miura_invert(space, û; λ = 0, seed = 1.0, tol = 1e-13, maxiter = 200)

Newton's method on ``\mathcal{M}_h^\lambda(\hat{v}) = \hat{u}``, returning `nothing` if it
does not converge.

At ``\lambda = 0`` the map is **neither surjective nor injective**. It is not surjective
because its image lies in ``C_{0,d} \le 0`` (see [`miura_map`](@ref)), and sharply because the
discrete Hill operator must be positive definite ([`hill_lambda0`](@ref)). Where a preimage
exists there are two — the two Floquet solutions of ``\psi'' = (-u - \lambda)\psi``, with
opposite signs of ``\int_\Omega v`` — and `seed` selects the branch.

The `λ` argument is what makes the map surjective in practice: a preimage exists exactly when
``\lambda < `` [`hill_lambda0`](@ref)`(space, û)`, which can always be arranged.
[`miura_lambda`](@ref) picks such a value. Returning `nothing` rather than throwing is
deliberate: at ``\lambda = 0`` there is genuinely no preimage for most `û`, and that is an
answer about the geometry of the map, not an error.
"""
function miura_invert(s::DiscreteSpace{T}, û::AbstractVector;
        λ::Real = 0, seed = one(T), tol = 1e-13,
        maxiter::Integer = 200) where {T}
    v = fill(convert(T, seed), nbasis(s))
    for _ in 1:maxiter
        r = miura_map(s, v, λ) .- û
        maximum(abs, r) < tol && return v
        L = miura_derivative(s, v)
        local Δ
        try
            Δ = L \ r
        catch
            return nothing
        end
        v = v .- Δ
        all(isfinite, v) || return nothing
    end
    return nothing
end

@doc raw"""
    miura_lambda(space, û; margin = 1//10)

A spectral parameter at which `û` **is** in the image of the Miura map:
``\lambda = \lambda_0 - \mathrm{margin} \cdot \max(1, |\lambda_0|)`` with
``\lambda_0`` = [`hill_lambda0`](@ref)`(space, û)`.

The criterion is ``\lambda < \lambda_0``, so any margin does; it is relative to
``|\lambda_0|`` because the preimage is *unbounded* as ``\lambda \to \lambda_0``, where
``\psi`` acquires a zero and ``v = \psi_x/\psi`` diverges. Sitting a fixed fraction below
the threshold keeps the Newton iteration of [`miura_invert`](@ref) well conditioned instead
of chasing the singular limit.

```jldoctest
julia> s = SplineSpace(20, 3);

julia> û = project(s, cosine(2π));            # zero mass: no preimage at λ = 0

julia> miura_invert(s, û) === nothing
true

julia> λ = miura_lambda(s, û); λ < hill_lambda0(s, û)
true

julia> v̂ = miura_invert(s, û; λ = λ); miura_map(s, v̂, λ) ≈ û
true
```
"""
miura_lambda(s::DiscreteSpace, û::AbstractVector; margin = 1//10) = (
    λ₀ = hill_lambda0(s, û); λ₀ - margin * max(one(λ₀), abs(λ₀)))

@doc raw"""
    bracket_directional(bracket, û, v)

The matrix ``D_{im} = \partial \left( \mathbb{P}(\hat{u}) \, v \right)_i /
\partial \hat{u}_m`` at fixed `v`.

This is the term that distinguishes the Jacobian of a Hamiltonian flow from the naive
``\mathbb{P} H''``: differentiating ``\mathbb{P}(\hat{u}) \, g(\hat{u})`` produces one term
from the gradient and one from the bracket, and this is the second.

It vanishes for a constant bracket, and is available in closed form for the others, so the
``O(N^3)`` derivative tensor of [`poisson_derivative`](@ref) is needed only for the
Jacobiator and never in a time loop.
"""
function bracket_directional(b::DiscreteBracket{T}, û::AbstractVector,
        v::AbstractVector) where {T}
    dP = poisson_derivative(b, û)
    N = length(û)
    D = zeros(T, N, N)
    @inbounds for m in 1:N, i in 1:N

        s = zero(T)
        for j in 1:N
            s += dP[m, i, j] * v[j]
        end
        D[i, m] = s
    end
    return D
end

function bracket_directional(b::ConstantBracket{T}, û::AbstractVector, v::AbstractVector) where {T}
    zeros(T, length(û), length(û))
end

function bracket_directional(b::MiuraBracket{T}, û::AbstractVector, v::AbstractVector) where {T}
    zeros(T, length(û), length(û))
end

function bracket_directional(b::AffineBracket, û::AbstractVector, v::AbstractVector)
    b.Minv * kernel_directional(b, b.Minv * v)
end

function bracket_directional(b::GaugedBracket{T}, û::AbstractVector,
        v::AbstractVector) where {T}
    gv = b.g.(û)
    dv = b.dg.(û)
    # (J v)_i = g_i Σ_j K_ij g_j v_j, so the derivative has a diagonal part from g_i and a
    # rank-structured part from g_j inside the sum
    D = Diagonal(dv .* (b.K * (gv .* v))) + Diagonal(gv) * b.K * Diagonal(dv .* v)
    Matrix(D)
end
