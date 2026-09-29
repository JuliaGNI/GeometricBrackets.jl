
#
# Dirac reduction of a Poisson tensor onto a constraint surface.
#
# Split the coordinates into a coarse block V1 (`keep`) and a constrained block V2 (`con`),
# and impose u_a = 0 for a in V2. When the constraint matrix C = J22 is invertible -- the
# constraints are SECOND CLASS -- the Dirac bracket
#
#     J*_ij = J_ij - J_ia (C^-1)_ab J_bj
#
# has the constraints as Casimirs, and its restriction to V1 is the Schur complement
# Jhat = J11 - J12 J22^-1 J21.
#
# Everything here is written once and generically in the element type. With `Rational{BigInt}`
# the identities come out exactly -- `iszero`, not "below 1e-12" -- and with `Float64` the same
# code runs the refinement studies. The Python prototypes carried an exact and a
# floating-point copy of each of these routines; keeping one is what stops the two drifting
# apart, which is a failure mode those prototypes actually recorded.
#
# SIGN CONVENTION OF THE JACOBIATOR. `jacobiator` uses the convention of `jacobi_residual`,
# sum_l (J_il d_l J_jk + cyclic), which for antisymmetric J is the negative of the form used
# in the Python prototypes. Every claim these tools serve is either "this vanishes" or "these
# two agree", so a global sign is invisible to all of them -- but it is worth knowing when
# comparing intermediate numbers against the Python.
#
# This was a file-level `@doc` block attached to no binding, which Documenter drops without a
# word: none of it reached the manual. The prose now lives in `docs/src/dirac.md`, where it is
# rendered; the comment stays so that a reader of the source has the definition to hand.

"""
    dirac_blocks(J, keep, con) -> (J11, J12, J21, J22)

The four blocks of `J` under the splitting into `keep` and `con`.
"""
function dirac_blocks(J::AbstractMatrix, keep, con)
    (J[keep, keep], J[keep, con], J[con, keep], J[con, con])
end

@doc raw"""
    schur_complement(J, keep, con)

The reduced bracket ``\hat{\mathbb{J}} = \mathbb{J}_{11} -
\mathbb{J}_{12}\mathbb{J}_{22}^{-1}\mathbb{J}_{21}`` on the coarse block.

Throws if ``\mathbb{J}_{22}`` is singular, which is exactly the statement that the
constraints are not second class and that Dirac reduction is not defined here.
"""
function schur_complement(J::AbstractMatrix, keep, con)
    J11, J12, J21, J22 = dirac_blocks(J, keep, con)
    return J11 - J12 * (J22 \ J21)
end

@doc raw"""
    dirac_tensor(J, con)

The full ``N \times N`` Dirac tensor ``\mathbb{J}^*_{ij} = \mathbb{J}_{ij} -
\mathbb{J}_{ia}(C^{-1})_{ab}\mathbb{J}_{bj}``, with ``C = \mathbb{J}_{22}``.

Its ``V_2`` rows and columns vanish identically, and the constraints ``u_a`` are Casimirs of
it — that is the point of the construction.
"""
function dirac_tensor(J::AbstractMatrix, con)
    C = J[con, con]
    return J - J[:, con] * (C \ J[con, :])
end

@doc raw"""
    dirac_R(J, con)

The Dirac projector ``R = I - \mathbb{J}[:, V_2] \, C^{-1} \, \Pi_{V_2}``, for which
``\mathbb{J}^* = R \mathbb{J} R^\top``.

``R^2 = R``, and the ``V_2`` rows vanish identically: for ``i, b \in V_2`` one has
``R^i{}_b = \delta^i_b - C_{ia}(C^{-1})_{ab} = 0``. Hence ``\operatorname{rank} R = \dim V_1``
and ``\dim \ker R = \dim V_2`` — and that kernel is precisely the room in which the
Jacobiator of ``\mathbb{J}`` can hide without ever reaching the reduced bracket. This is why
Dirac reduction *preserves* the Jacobi identity but never *creates* it.
"""
function dirac_R(J::AbstractMatrix{T}, con) where {T}
    C = J[con, con]
    R = Matrix{T}(I, size(J, 1), size(J, 1))
    R[:, con] -= J[:, con] * inv(C)
    return R
end

"""
    dirac_projector(J, keep, con)

``P = \\Pi_{V_1} R``, the `length(keep) × N` matrix whose rows are the `keep` rows of
[`dirac_R`](@ref). This is the tensor through which the Jacobiator transports.
"""
dirac_projector(J::AbstractMatrix, keep, con) = dirac_R(J, con)[keep, :]

@doc raw"""
    dschur(C3, u, keep, con, l)

``\partial \hat{\mathbb{J}} / \partial u_l`` on the constraint surface, exactly, for a
bracket linear in the field with structure constants `C3[m,i,j]`.

With ``A = \partial\mathbb{J}/\partial u_l`` constant and ``B = \mathbb{J}_{22}^{-1}``,
differentiating the Schur complement and using ``\partial B = -BA_{22}B`` gives the four
terms

```math
\partial_l \hat{\mathbb{J}} = A_{11} - A_{12}B\mathbb{J}_{21}
    + \mathbb{J}_{12}BA_{22}B\mathbb{J}_{21} - \mathbb{J}_{12}BA_{21} .
```

`l` must lie in `keep`. Differentiating along a constrained direction does not commute with
restricting to the surface, and is not what the Jacobiator of the reduced bracket asks for.
"""
function dschur(C3::AbstractArray{T, 3}, u::AbstractVector, keep, con, l::Int) where {T}
    l ∈ keep || throw(ArgumentError("differentiate along kept directions only, got l = $l"))
    J = lie_poisson_matrix(C3, u)
    _, J12, J21, J22 = dirac_blocks(J, keep, con)
    B = inv(J22)
    A = C3[l, :, :]
    A11, A12, A21, A22 = dirac_blocks(A, keep, con)
    return A11 - A12 * B * J21 + J12 * B * A22 * B * J21 - J12 * B * A21
end

"""
    reduced_bracket(C3, u, keep, con) -> (Ĵ, dĴ)

The reduced bracket and its derivative tensor on the constraint surface, in the
`dĴ[l,i,j]` convention of [`poisson_derivative`](@ref).

`u` must be a point with `u[a] == 0` for `a in con`. Both outputs are indexed by position
*within* `keep`, so the pair goes straight into [`jacobi_residual`](@ref).
"""
function reduced_bracket(C3::AbstractArray{T, 3}, u::AbstractVector, keep, con) where {T}
    J = lie_poisson_matrix(C3, u)
    Ĵ = schur_complement(J, keep, con)
    nk = length(keep)
    dĴ = Array{eltype(Ĵ), 3}(undef, nk, nk, nk)
    for (pos, l) in enumerate(keep)
        dĴ[pos, :, :] = dschur(C3, u, keep, con, l)
    end
    return Ĵ, dĴ
end

@doc raw"""
    jacobiator(J, dJ)

The Jacobiator trivector ``[\mathbb{J},\mathbb{J}]^{ijk} = \sum_l (\mathbb{J}_{il}
\partial_l \mathbb{J}_{jk} + \text{cyclic})``, as a full `N × N × N` array.

Totally antisymmetric, so only the `i < j < k` entries carry information — but the full
array is what [`project_jacobiator`](@ref) has to contract, and at the sizes these
structural tests run at there is no reason to store less.

[`jacobi_residual`](@ref) is `maximum(abs, jacobiator(J, dJ))` up to its normalisation.
"""
function jacobiator(J::AbstractMatrix, dJ::AbstractArray{T, 3}) where {T}
    N = size(J, 1)
    E = promote_type(eltype(J), T)
    A = zeros(E, N, N, N)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(E)
        for l in 1:N
            s += J[i, l] * dJ[l, j, k]
        end
        A[i, j, k] = s
    end
    B = similar(A)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        B[i, j, k] = A[i, j, k] + A[j, k, i] + A[k, i, j]
    end
    return B
end

@doc raw"""
    project_jacobiator(BJ, P)

``P^i{}_l P^j{}_m P^k{}_n [\mathbb{J},\mathbb{J}]^{lmn}`` — the Jacobiator of the ambient
bracket pushed through the Dirac projector of [`dirac_projector`](@ref).

The central identity of the Dirac companion note is that this equals the Jacobiator of the
reduced bracket:

```math
[\hat{\mathbb{J}},\hat{\mathbb{J}}] = P^{\otimes 3} [\mathbb{J},\mathbb{J}] .
```

It transports *tensorially*. So the reduced bracket is Poisson whenever the ambient one is,
and — the point — Dirac reduction cannot repair a Jacobiator that was already nonzero
unless the projector happens to annihilate it.
"""
function project_jacobiator(BJ::AbstractArray{T, 3}, P::AbstractMatrix) where {T}
    nk, N = size(P)
    E = promote_type(T, eltype(P))
    # contract one index at a time: O(nk N^3) rather than the O(nk^3 N^3) of the direct sum
    S1 = zeros(E, nk, N, N)
    @inbounds for n in 1:N, m in 1:N, i in 1:nk
        s = zero(E)
        for l in 1:N
            s += P[i, l] * BJ[l, m, n]
        end
        S1[i, m, n] = s
    end
    S2 = zeros(E, nk, nk, N)
    @inbounds for n in 1:N, j in 1:nk, i in 1:nk
        s = zero(E)
        for m in 1:N
            s += P[j, m] * S1[i, m, n]
        end
        S2[i, j, n] = s
    end
    S3 = zeros(E, nk, nk, nk)
    @inbounds for k in 1:nk, j in 1:nk, i in 1:nk
        s = zero(E)
        for n in 1:N
            s += P[k, n] * S2[i, j, n]
        end
        S3[i, j, k] = s
    end
    return S3
end

"""
    restrict_c(C3, keep)

The structure constants of the *naive* truncation to `keep` — simply the corresponding
sub-block of `C3`.

The comparison against [`reduced_bracket`](@ref): the naive truncation throws the
constrained directions away, Dirac reduction eliminates them. They agree only when ``V_2``
is an ideal.
"""
restrict_c(C3::AbstractArray{T, 3}, keep) where {T} = C3[keep, keep, keep]

"""
    is_antisymmetric_c(C3)

Whether ``c_{ij}^m = -c_{ji}^m`` exactly.
"""
function is_antisymmetric_c(C3::AbstractArray{T, 3}) where {T}
    all(C3[m, i, j] == -C3[m, j, i]
    for m in axes(C3, 1), i in axes(C3, 2), j in axes(C3, 3))
end

"""
    is_ideal(C3, con, keep)

Whether ``V_2`` is an ideal, i.e. ``[\\,\\cdot\\,, V_2]`` has no ``V_1`` component.

When it is, the naive truncation of [`restrict_c`](@ref) is already a Lie algebra and Dirac
reduction has nothing to add. The interesting cases are the ones where it is not.
"""
function is_ideal(C3::AbstractArray{T, 3}, con, keep) where {T}
    all(iszero(C3[m, i, a]) for i in axes(C3, 2), a in con, m in keep)
end

_nonsingular(A::AbstractMatrix{<:Union{Rational, Integer}}, _) = !iszero(det(A))
_nonsingular(A::AbstractMatrix, tol) = rank(A; atol = tol) == size(A, 1)

@doc raw"""
    maximal_second_class(J, con; tol = nothing)

A maximal subset `S ⊆ con` on which the constraint matrix ``\mathbb{J}[S,S]`` is invertible,
returned sorted.

The diagonal of an antisymmetric matrix vanishes, so constrained directions are only useful
in **pairs**: an odd-sized block is always singular. The set is therefore grown two indices
at a time, greedily, while the block stays nonsingular. That parity is not a technicality —
it is why a constrained block of odd dimension can never be fully second class, and it is
one of the two obstructions the Dirac note raises before any computation is done.

Exact element types use `det ≠ 0`; floating-point ones a rank test at `tol`, defaulting to
`1e-8` relative to the largest entry of `J`.
"""
function maximal_second_class(J::AbstractMatrix{T}, con; tol = nothing) where {T}
    atol = tol === nothing ? 1e-8 * float(maximum(abs, J)) : tol
    sel = Int[]
    rest = collect(con)
    while true
        grew = false
        for ia in eachindex(rest), ib in (ia + 1):lastindex(rest)

            trial = vcat(sel, rest[ia], rest[ib])
            if _nonsingular(J[trial, trial], atol)
                sel = trial
                deleteat!(rest, (ia, ib))
                grew = true
                break
            end
        end
        grew || return sort(sel)
    end
end
