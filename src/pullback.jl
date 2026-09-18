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

because ``\nabla_x = J^{-T} \hat\nabla``. This type holds ``m``, ``|\det J|``, ``\mathbb{D}``
and ``J^{-T}`` at the quadrature nodes, and the physical coordinates ``F(\hat{x}_q)`` that a
physical field is sampled at.

# Why one object rather than three call sites

A mapped assembly needs the same geometry in several places — the measure a metric bracket
integrates against, the tensor coefficient of the weighted stiffness, the frame a gradient is
read in, and any diagnostic that integrates over the domain. Supplying it separately to each
is how a factor ends up in two of them and not the third, which is a wrong answer rather than
a failed assertion. Here they are computed once, from one map and one density, and read off:

```julia
pb = PulledBack(space, F, DF; density = ρ)
K  = tensor_weighted_matrix(space, metric(pb))        # ∫ ∇u·∇v dμ, pulled back
G  = CollisionBracket(space, Λ, pb)                   # measure, frame and pairing together
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
    volume::Vector{T}
    metric::Matrix{Vector{T}}
    frame::Matrix{Vector{T}}

    function PulledBack(s::DiscreteSpace{T}, F, DF;
            density = nothing, tensor = nothing) where {T}
        x̂ = quadrature_nodes(s)
        D = length(first(x̂))
        Q = length(x̂)

        x = [NTuple{D, T}(F(pt)) for pt in x̂]
        ρ = _density_vector(density, x, Q)

        m = Vector{T}(undef, Q)
        v = Vector{T}(undef, Q)
        𝔻 = [Vector{T}(undef, Q) for _ in 1:D, _ in 1:D]
        𝔽 = [Vector{T}(undef, Q) for _ in 1:D, _ in 1:D]

        # the identity serves twice, as the default coefficient and as the right-hand side the
        # frame is solved for. Both are loop-invariant, so it is built once rather than at each
        # of the Q nodes; `\` copies its right-hand side, so the one matrix is not consumed
        A₀ = Matrix{T}(I, D, D)

        for q in 1:Q
            J = _jacobian_matrix(DF(x̂[q]), D)
            detJ = abs(det(J))
            v[q] = detJ
            m[q] = ρ[q] * detJ

            # m J⁻¹ A J⁻ᵀ, formed as a solve rather than an explicit inverse: at D = 2 or 3
            # the difference is not the arithmetic but that `\` is the one spelling which
            # says what is meant and cannot be transposed by accident.
            A = tensor === nothing ? A₀ : Matrix{T}(tensor(x[q]))
            G = m[q] * (J \ A) / J'

            # J⁻ᵀ itself, which a caller that differentiates rather than integrates needs:
            # the metric above has the measure and the coefficient folded in and cannot be
            # taken apart again.
            invJᵀ = J' \ A₀

            for k in 1:D, l in 1:D

                𝔻[k, l][q] = G[k, l]
                𝔽[k, l][q] = invJᵀ[k, l]
            end
        end

        new{T, D}(x, m, v, 𝔻, 𝔽)
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
of `quadrature_weights`.

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
    volume_element(pb::PulledBack)

The volume element ``|\det J|`` at the quadrature nodes — the **plain** physical measure
``dx`` against the parameter quadrature, with the density left out.

[`measure`](@ref) is ``\rho \, |\det J|`` and cannot be divided back down: ``\rho`` may
vanish, and a caller that needs both weights should not be reconstructing one from the other.
The two are different integrals, and the Grad-Shafranov discretisation needs both — the
bracket integrates against ``d\mu = \rho \, dx`` while the pairing that defines a functional
derivative is against ``dx``.
"""
volume_element(pb::PulledBack) = pb.volume

@doc raw"""
    frame(pb::PulledBack)

The inverse transpose Jacobian ``J^{-T}`` at the quadrature nodes, as the ``D \times D``
matrix of per-node vectors.

This is the matrix that carries a parameter gradient to the physical one,
``\nabla_x = J^{-T} \hat\nabla``. A space's derivative tables are parameter derivatives, so
anything that reads a *direction* rather than integrating a scalar — a perpendicular, a
rotation, a cross product — needs this and is wrong without it. [`metric`](@ref) does not
serve: it has the measure and the physical coefficient folded in, and a congruence cannot be
taken apart again.
"""
frame(pb::PulledBack) = pb.frame

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

The largest discrepancy between the supplied Jacobian `DF` and a central difference of `F`,
over the parameter points `x̂`, taken **relative to the difference where that exceeds one and
absolute where it does not**.

The mixed measure is deliberate: a map's Jacobian has entries that pass through zero — the
polar chart's ``\partial r / \partial \theta`` does, at two angles out of every four — and a
relative comparison there divides by round-off and reports a failure that is not one.

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
