@doc raw"""
    PulledBack(space, F, DF; density = nothing, tensor = nothing)

The pullback of a physical domain's measure and metric onto the parameter domain a
[`DiscreteSpace`](@ref) is built on, evaluated once at the quadrature nodes.

A map ``F : \hat{\Omega} \to \Omega`` with Jacobian ``J = \partial F / \partial \hat{x}``
carries an integral against ``d\mu = \rho(x) \, dx`` to

```math
\int_\Omega f \, d\mu
    = \int_{\hat\Omega} f \circ F \; \underbrace{\rho(F) \, |\det J|}_{m} \; d\hat{x} ,
```

and a Dirichlet form against a physical coefficient ``\mathbb{A}`` to

```math
\int_\Omega (\nabla_x u)^T \mathbb{A} \, (\nabla_x v) \, d\mu
    = \int_{\hat\Omega} (\hat\nabla u)^T
      \underbrace{\big[\, m \, J^{-1} \mathbb{A} J^{-T} \,\big]}_{\mathbb{D}}
      (\hat\nabla v) \, d\hat{x} ,
```

because ``\nabla_x = J^{-T} \hat\nabla``. This type holds ``m`` and ``\mathbb{D}`` at the
quadrature nodes, and the physical coordinates ``F(\hat{x}_q)`` that a physical field is
sampled at.

# Why one object rather than three call sites

A mapped assembly needs the same weight in three places — the measure a metric bracket
integrates against, the tensor coefficient of the weighted stiffness, and any diagnostic that
integrates over the domain. Supplying it separately to each is how a factor ends up in two of
them and not the third, which is a wrong answer rather than a failed assertion. Here they are
computed once, from one map and one density, and read off:

```julia
pb = PulledBack(space, F, DF; density = ρ)
K  = tensor_weighted_matrix(space, metric(pb))        # ∫ ∇u·∇v dμ, pulled back
G  = CollisionBracket(space, Λ; density = measure(pb))
∫f = dot(quadrature_weights(space) .* measure(pb), f) # any integral over the domain
```

# The Jacobian is supplied, not differenced

`DF` returns the ``D \times D`` Jacobian at a parameter point and is a required argument. A
finite difference of `F` would put a truncation error inside every matrix assembled from it,
and the maps this exists for are written down in closed form — there is nothing to
approximate. [`jacobian_residual`](@ref) checks a supplied `DF` against a central difference
of `F`, which is the cheap guard against a slip in the algebra; it is a test, not a fallback.

# Arguments

  - `F` maps a parameter point, a `D`-tuple, to a physical point, a `D`-tuple.
  - `DF` maps a parameter point to the ``D \times D`` Jacobian, `DF(x̂)[i,j] = ∂Fᵢ/∂x̂ⱼ`.
  - `density` is ``\rho``, a function of the **physical** point or a vector of samples, and
    defaults to one. A Grad-Shafranov weight ``d\mu = dr \, dz / r`` is `x -> 1 / x[1]`, in
    the physical coordinates it is written in.
  - `tensor` is ``\mathbb{A}``, a function of the physical point returning a ``D \times D``
    matrix, and defaults to the identity — the plain Laplacian.

`|\det J|` is taken in absolute value: the measure is a measure, and a map that reverses
orientation is not an error.
"""
struct PulledBack{T, D}
    nodes::Vector{NTuple{D, T}}
    measure::Vector{T}
    metric::Matrix{Vector{T}}

    function PulledBack(s::DiscreteSpace{T}, F, DF;
            density = nothing, tensor = nothing) where {T}
        x̂ = quadrature_nodes(s)
        D = length(first(x̂))
        Q = length(x̂)

        x = [NTuple{D, T}(F(pt)) for pt in x̂]
        ρ = _density_vector(density, x, Q)

        m = Vector{T}(undef, Q)
        𝔻 = [Vector{T}(undef, Q) for _ in 1:D, _ in 1:D]

        for q in 1:Q
            J = _jacobian_matrix(DF(x̂[q]), D)
            detJ = abs(det(J))
            m[q] = ρ[q] * detJ

            # m J⁻¹ A J⁻ᵀ, formed as a solve rather than an explicit inverse: at D = 2 or 3
            # the difference is not the arithmetic but that `\` is the one spelling which
            # says what is meant and cannot be transposed by accident.
            A = tensor === nothing ? Matrix{T}(I, D, D) : Matrix{T}(tensor(x[q]))
            G = m[q] * (J \ A) / J'

            for k in 1:D, l in 1:D

                𝔻[k, l][q] = G[k, l]
            end
        end

        new{T, D}(x, m, 𝔻)
    end
end

function _jacobian_matrix(J::AbstractMatrix, D::Integer)
    size(J) == (D, D) ? J :
    throw(DimensionMismatch(
        "the Jacobian is $(size(J)) but the space is $(D)-dimensional"))
end

_density_vector(::Nothing, x, Q) = ones(eltype(first(x)), Q)

function _density_vector(ρ::AbstractVector, x, Q)
    length(ρ) == Q || throw(DimensionMismatch(
        "the density was sampled at $(length(ρ)) points but the quadrature grid has $(Q)"))
    ρ
end

_density_vector(ρ, x, Q) = [ρ(pt) for pt in x]

"""
    nodes(pb::PulledBack)

The **physical** coordinates ``F(\\hat{x}_q)`` of the quadrature nodes, in the flattened order
of [`quadrature_weights`](@ref).

This is what a physical field is sampled at. A mobility written in physical coordinates — the
Grad-Shafranov ``M = Cr^2 + D`` — is evaluated here and not at the parameter nodes, and the
two are not interchangeable.
"""
nodes(pb::PulledBack) = pb.nodes

@doc raw"""
    measure(pb::PulledBack)

The pulled-back measure density ``m = \rho(F) \, |\det J|`` at the quadrature nodes.

This is a **density against the parameter quadrature**, not a set of weights: an integral over
the domain is `dot(quadrature_weights(space) .* measure(pb), f)`. It is the vector to hand a
[`CollisionBracket`](@ref) as its `density`.
"""
measure(pb::PulledBack) = pb.measure

@doc raw"""
    metric(pb::PulledBack)

The pulled-back tensor coefficient ``\mathbb{D} = m \, J^{-1} \mathbb{A} J^{-T}``, as the
``D \times D`` matrix of per-node vectors that [`tensor_weighted_matrix`](@ref) takes.

``\mathbb{D}`` is symmetric wherever ``\mathbb{A}`` is, and positive semi-definite wherever
``\mathbb{A}`` is, since ``J^{-1} \mathbb{A} J^{-T}`` is a congruence and ``m \ge 0``. So a
mapped assembly inherits the definiteness of the physical coefficient and cannot lose it to
the map — which is worth knowing, because it means a metric bracket that goes indefinite on a
mapped domain has a coefficient problem and not a geometry problem.
"""
metric(pb::PulledBack) = pb.metric

Base.ndims(::PulledBack{T, D}) where {T, D} = D
Base.eltype(::PulledBack{T}) where {T} = T
Base.length(pb::PulledBack) = length(pb.measure)

function Base.show(io::IO, pb::PulledBack{T, D}) where {T, D}
    print(io, "PulledBack{", T, ", ", D, "} over ", length(pb), " quadrature nodes, ")
    print(io, "measure in [", minimum(pb.measure), ", ", maximum(pb.measure), "]")
end

@doc raw"""
    jacobian_residual(F, DF, x̂; h = 1e-6)

The largest relative discrepancy between the supplied Jacobian `DF` and a central difference
of `F`, over the parameter points `x̂`.

The guard against a slip in the algebra of a hand-derived Jacobian. It is a **check**, and the
difference is never used in an assembly: a truncation error of order ``h^2`` inside a stiffness
matrix is a wrong operator, not an approximate one.

```jldoctest
julia> F(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]));

julia> DF(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])];

julia> pts = [(r, θ) for r in 0.5:0.25:1.5, θ in range(0, 2π; length = 9)][:];

julia> jacobian_residual(F, DF, pts) < 1e-8
true
```
"""
function jacobian_residual(F, DF, x̂::AbstractVector; h = 1e-6)
    worst = 0.0
    for pt in x̂
        D = length(pt)
        J = DF(pt)
        for j in 1:D
            e = ntuple(i -> i == j ? h : zero(h), D)
            plus = F(ntuple(i -> pt[i] + e[i], D))
            minus = F(ntuple(i -> pt[i] - e[i], D))
            for i in 1:D
                fd = (plus[i] - minus[i]) / 2h
                worst = max(worst, abs(J[i, j] - fd) / max(1, abs(fd)))
            end
        end
    end
    return worst
end
