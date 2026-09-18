@doc raw"""
    PolarSplineSpace(quadrature)
    PolarSplineSpace(basis; nq, dmax)
    PolarSplineSpace(radial, angular; kwargs...)
    PolarSplineSpace(cells, p; kwargs...)

A two-dimensional [`DiscreteSpace`](@ref) on a parameter square whose left radial edge is a
**pole** — one point of the physical domain, reached from every angle, as on a mapped disk.

Wraps SimpleSplines' `PolarSplineBasis`, in which the first two radial rows are replaced by
three functions spanning the constants and the two linear functions of a pseudo-Cartesian
chart at the pole, so that the space is ``C^0`` and ``C^1`` there **by construction**. A
tensor-product space on such a square is not even ``C^0``: the map collapses the whole circle
``s = 0`` to a point and nothing constrains a tensor-product basis's ``\theta``-dependence on
it.

```jldoctest
julia> s = PolarSplineSpace((8, 16), 3);

julia> nbasis(s), ndims(s)
(147, 2)

julia> û = project(s, x -> 1.0);

julia> abs(evaluate(s, û, (0.0, 0.7)) - 1) < 1e-11
true
```

# What it shares with `TensorSplineSpace`, and what it does not

The interface is the same, and every generic assembly of `spaces.jl` runs on it unchanged,
because the three things they are written against have the same shapes: `basis_values(s, d)`
is a sparse ``N \times Q`` table over the flattened quadrature grid, `quadrature_weights(s)`
is the matching flat vector, and `mass_factorization(s)` answers `\`. As there, `d` is a
per-axis multi-index and a scalar `d ≥ 1` is rejected rather than resolved to an axis.

Three things differ, all of them consequences of a pole function reaching around the whole
angular axis:

  - **The index set is not a product.** There is no `size(s)` and no `nodes(s)`, and a
    coefficient vector is not reshaped into an array anywhere — `evaluate` takes the vector as
    it is. The three pole functions belong to no radial index and to no angular one, so
    neither a shape nor a per-degree-of-freedom coordinate exists. The flattening still runs
    the radial axis fastest, which is the convention of every table here.
  - **There is no `KroneckerMass`.** `mass_factorization` is a sparse Cholesky, so a solve is
    ``O(N^{3/2})`` rather than ``D`` one-dimensional solves. Measured at ``64 \times 128``
    cubic cells: 0.74 ms against 0.136 ms for the tensor-product space at the same mesh, and
    under a millisecond either way.
  - **[`inverse_mass_matrix`](@ref) has no Kronecker shortcut** and is an ``N \times N`` dense
    solve. It exists because the generic assemblies name it; it is the wrong thing to call in
    a loop, and nothing here calls it.

# The measure is the parameter square's

Every matrix this space assembles integrates against ``ds \, d\theta``, exactly as
[`TensorSplineSpace`](@ref) integrates against ``dx``. A polar space is used for a *mapped*
domain, whose measure and metric are the map's and not the space's, and they are supplied
through [`PulledBack`](@ref) — `metric(pb)` to [`tensor_weighted_matrix`](@ref), `measure(pb)`
to a bracket's `density`, and `nodes(pb)` wherever a physical coefficient is sampled.

Keeping the measure out of the space is what makes the two spaces mean the same thing by the
same method names. What keeps the three call sites consistent is that they all read one
`PulledBack`.
"""
struct PolarSplineSpace{T, QT <: PolarSplineQuadrature{T}} <: DiscreteSpace{T}
    quadrature::QT
    x::Vector{NTuple{2, T}}

    function PolarSplineSpace(q::PolarSplineQuadrature{T}) where {T}
        # `Iterators.product` runs its first factor fastest, which is the radial axis, and is
        # the order of `quadrature_weights(q)` and of every column of `basis_values(q, d)`.
        x = vec([NTuple{2, T}(pt) for pt in Iterators.product(quadrature_nodes(q)...)])
        new{T, typeof(q)}(q, x)
    end
end

function PolarSplineSpace(B::PolarSplineBasis; kwargs...)
    PolarSplineSpace(PolarSplineQuadrature(B; kwargs...))
end

function PolarSplineSpace(radial::AbstractBSplineBasis, angular::AbstractBSplineBasis;
        kwargs...)
    PolarSplineSpace(PolarSplineBasis(radial, angular); kwargs...)
end

@doc raw"""
    PolarSplineSpace(cells::NTuple{2,Integer}, p; L = 2π, kwargs...)

The space on ``[0,1] \times [0,L)`` with `cells[1]` radial and `cells[2]` angular cells, both
of degree `p` — the radial axis clamped, the angular axis periodic, which a pole leaves no
choice about.

`p` may be given per axis as a tuple. The radial degree must be at least two, which is what
makes the pole triangle a ``C^1`` construction rather than a ``C^0`` one.
"""
function PolarSplineSpace(cells::NTuple{2, Integer}, p; L = 2π, kwargs...)
    ps = _per_axis(p, 2, "the degree")
    PolarSplineSpace(BSplineBasis(UniformMesh(cells[1], zero(L) .. one(L)), ps[1]),
        PeriodicBSplineBasis(UniformMesh(cells[2], zero(L) .. L), ps[2]); kwargs...)
end

quadrature(s::PolarSplineSpace) = s.quadrature
basis(s::PolarSplineSpace) = basis(s.quadrature)
nbasis(s::PolarSplineSpace) = nbasis(s.quadrature)
degree(s::PolarSplineSpace) = degree(basis(s))
order(s::PolarSplineSpace) = order(basis(s))
ncells(s::PolarSplineSpace) = ncells(basis(s))
Base.ndims(::PolarSplineSpace) = 2

"""
    pole(space::PolarSplineSpace)
    pole_triangle(space::PolarSplineSpace)
    pseudo_cartesian(space::PolarSplineSpace, x)

Forwarded to the underlying `PolarSplineBasis`: the radial coordinate of the pole, the three
pole-triangle vertices, and the chart in which the space is ``C^1`` there.
"""
pole(s::PolarSplineSpace) = pole(basis(s))
pole_triangle(s::PolarSplineSpace) = pole_triangle(basis(s))
pseudo_cartesian(s::PolarSplineSpace, x) = pseudo_cartesian(basis(s), x)

"""
    quadrature_nodes(space::PolarSplineSpace)

The quadrature nodes as a flat vector of `(s, θ)` pairs, the radial axis running fastest.

A vector of points rather than the per-axis tuple the underlying quadrature holds, because
that is what [`project`](@ref), `field` and every generic assembly of `spaces.jl` index
against, and what a coefficient is sampled at.
"""
quadrature_nodes(s::PolarSplineSpace) = s.x
quadrature_weights(s::PolarSplineSpace) = quadrature_weights(s.quadrature)
mass_operator(s::PolarSplineSpace) = mass_operator(s.quadrature)
mass_factorization(s::PolarSplineSpace) = mass_operator(s.quadrature)
mass_matrix(s::PolarSplineSpace) = mass_matrix(s.quadrature)
basis_integrals(s::PolarSplineSpace) = basis_integrals(s.quadrature)

"""
    domainlength(space::PolarSplineSpace)

The per-axis lengths of the **parameter** square, as a tuple. The extent of the mapped domain
is the map's business, not the space's; see [`domainvolume`](@ref) for the scalar.
"""
domainlength(s::PolarSplineSpace) = map(domainlength, bases(basis(s)))
domainvolume(s::PolarSplineSpace) = prod(domainlength(s))

@doc raw"""
    basis_values(space::PolarSplineSpace, d::NTuple{2,Int})
    basis_values(space::PolarSplineSpace, d::Integer = 0)

The table ``\Phi_d[K,R] = \partial_s^{d_1} \partial_\theta^{d_2} \Psi_K(x_R)`` over the
flattened quadrature grid, sparse, memoised by the underlying quadrature.

`d` is a **per-axis multi-index**. A scalar `d ≥ 1` names no derivative on a two-dimensional
space and is rejected rather than resolved to one of the axes, exactly as for
[`TensorSplineSpace`](@ref).
"""
basis_values(s::PolarSplineSpace, d::NTuple{2, Int}) = basis_values(s.quadrature, d)
basis_values(s::PolarSplineSpace, d::Integer = 0) = basis_values(s.quadrature, d)

"""
    mixed_matrix(space::PolarSplineSpace, a::NTuple{2,Int}, b::NTuple{2,Int})

The matrix ``\\int D^a \\Psi_K \\, D^b \\Psi_L \\, ds \\, d\\theta``, memoised by the
underlying quadrature. Against the **parameter** measure; a mapped one goes through
[`weighted_matrix`](@ref) or [`tensor_weighted_matrix`](@ref).
"""
function mixed_matrix(s::PolarSplineSpace, a::NTuple{2, Int}, b::NTuple{2, Int})
    mixed_matrix(s.quadrature, a, b)
end

@doc raw"""
    weighted_matrix(space::PolarSplineSpace, f, a::NTuple{2,Int}, b::NTuple{2,Int})

The matrix ``\int f(x) \, D^a \Psi_K \, D^b \Psi_L \, ds \, d\theta``, with `f` either a
function of the coordinate pair or a vector already sampled on the flattened quadrature grid.

Not memoised, unlike [`mixed_matrix`](@ref): the coefficient of a mapped assembly or of a
metric bracket changes at every Newton iteration. This is where the measure of a mapped
domain enters — `measure(pb)` of a [`PulledBack`](@ref) is exactly such a coefficient.
"""
function weighted_matrix(s::PolarSplineSpace, f, a::NTuple{2, Int}, b::NTuple{2, Int})
    weighted_matrix(s, map(f, quadrature_nodes(s)), a, b)
end

function weighted_matrix(s::PolarSplineSpace, f::AbstractVector, a::NTuple{2, Int},
        b::NTuple{2, Int})
    length(f) == length(quadrature_weights(s)) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(f)) points but the quadrature grid has " *
        "$(length(quadrature_weights(s)))"))
    basis_values(s, a) * Diagonal(f .* quadrature_weights(s)) * basis_values(s, b)'
end

"""
    derivative_matrix(space::PolarSplineSpace, k)

The matrix ``\\int \\Psi_K \\, \\partial_k \\Psi_L \\, ds \\, d\\theta``.

There is no axis-free form: ``\\partial_s`` and ``\\partial_\\theta`` are different operators,
so the scalar-index form inherited from [`DiscreteSpace`](@ref) raises rather than picking one.
"""
function derivative_matrix(s::PolarSplineSpace, k::Integer)
    mixed_matrix(s, (0, 0), _unit_index(2, k))
end

"""
    stiffness_matrix(space::PolarSplineSpace)

The matrix ``\\int \\nabla \\Psi_K \\cdot \\nabla \\Psi_L \\, ds \\, d\\theta`` over the
parameter square, symmetric positive semi-definite with the constants in its kernel.
"""
function stiffness_matrix(s::PolarSplineSpace)
    sum(mixed_matrix(s, _unit_index(2, k), _unit_index(2, k)) for k in 1:2)
end

@doc raw"""
    tensor_weighted_matrix(space::PolarSplineSpace, 𝔻)

The matrix ``\mathbb{A}_{KL} = \int \partial_k \Psi_K \, \mathbb{D}_{kl} \, \partial_l \Psi_L
\, ds \, d\theta``, summed over the two axes — the operator a mapped assembly is built from,
and the one [`metric`](@ref)`(::PulledBack)` produces the coefficient for.

The coefficient forms are those of [`tensor_weighted_matrix`](@ref)`(::TensorSplineSpace, 𝔻)`.
"""
function tensor_weighted_matrix(s::PolarSplineSpace,
        𝔻::AbstractMatrix{<:AbstractVector})
    size(𝔻) == (2, 2) || throw(DimensionMismatch(
        "the coefficient is $(size(𝔻)) but the space is 2-dimensional"))
    sum(weighted_matrix(s, 𝔻[k, l], _unit_index(2, k), _unit_index(2, l))
    for k in 1:2, l in 1:2)
end

function tensor_weighted_matrix(s::PolarSplineSpace{T},
        𝔻::AbstractMatrix{<:Number}) where {T}
    size(𝔻) == (2, 2) || throw(DimensionMismatch(
        "the coefficient is $(size(𝔻)) but the space is 2-dimensional"))
    A = spzeros(T, nbasis(s), nbasis(s))
    for k in 1:2, l in 1:2

        iszero(𝔻[k, l]) && continue
        A += 𝔻[k, l] * mixed_matrix(s, _unit_index(2, k), _unit_index(2, l))
    end
    return A
end

function tensor_weighted_matrix(s::PolarSplineSpace,
        𝔻::AbstractVector{<:AbstractMatrix})
    length(𝔻) == length(quadrature_weights(s)) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(𝔻)) points but the quadrature grid has " *
        "$(length(quadrature_weights(s)))"))
    tensor_weighted_matrix(s, [[𝔻[q][k, l] for q in eachindex(𝔻)] for k in 1:2, l in 1:2])
end

function tensor_weighted_matrix(s::PolarSplineSpace, 𝔻)
    tensor_weighted_matrix(s, map(𝔻, quadrature_nodes(s)))
end

@doc raw"""
    inverse_mass_matrix(space::PolarSplineSpace)

The dense inverse mass matrix ``\mathbb{M}^{-1}``, formed **on every call** and not stored.

There is no Kronecker shortcut here — the pole rows are what break it — so this is an
``N \times N`` solve against the sparse Cholesky, and the storage is ``N^2``: at
``N = 8323`` that is 554 MB. It exists because the generic assemblies of `spaces.jl` name it,
and it is the wrong thing to call in a loop. `mass_factorization` returns the
factorisation, and `\`, `mass_solve!`, [`project`](@ref) and [`project!`](@ref) all go through
it without forming this matrix.
"""
function inverse_mass_matrix(s::PolarSplineSpace{T}) where {T}
    mass_factorization(s) \ Matrix{T}(I, nbasis(s), nbasis(s))
end

@doc raw"""
    evaluate(space::PolarSplineSpace, û, x, d = (0, 0))

The mixed derivative ``\partial_s^{d_1} \partial_\theta^{d_2} u_h`` of the field with
coefficient vector `û` at the parameter point `x`, or at each point of a vector of points.

`û` is the flat vector of length `nbasis`, and unlike [`TensorSplineSpace`](@ref) it is
**not** reshaped: the polar index set is not a product, so there is no array shape to reshape
it into.
"""
function evaluate(s::PolarSplineSpace, û::AbstractVector, x, d::NTuple{2, Int} = (0, 0))
    length(û) == nbasis(s) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the space has $(nbasis(s))"))
    evaluate(basis(s), û, x, d)
end

function evaluate(s::PolarSplineSpace, û::AbstractVector,
        X::AbstractVector{<:Union{Tuple, AbstractVector}}, d::NTuple{2, Int} = (0, 0))
    [evaluate(s, û, x, d) for x in X]
end

"""
    field(space::PolarSplineSpace, û, d::NTuple{2,Int})

The mixed derivative `d` of ``u_h`` sampled on the flattened quadrature grid — the operation
every variable-coefficient assembly starts from.
"""
function field(s::PolarSplineSpace, û::AbstractVector, d::NTuple{2, Int})
    basis_values(s, d)' * û
end

function Base.show(io::IO, s::PolarSplineSpace{T}) where {T}
    print(io, "PolarSplineSpace{", T, "}(", nbasis(s), " functions, ")
    print(io, "s: p=", degree(s)[1], ", n=", ncells(s)[1])
    print(io, " ⊕ θ: p=", degree(s)[2], ", n=", ncells(s)[2], ")")
end

@doc raw"""
    PlanarSplineSpace{T}

The spline spaces on a two-dimensional domain: a [`TensorSplineSpace`](@ref) with two axes, or
a [`PolarSplineSpace`](@ref).

A union rather than an abstract type, because the two share no supertype below
[`DiscreteSpace`](@ref) — `TensorSplineSpace` is generic in its dimension and cannot subtype a
two-dimensional-only abstraction — and because `DiscreteSpace` carries no dimension parameter
to constrain. It is what a bracket whose *algebra* is planar dispatches on: a
[`CollisionBracket`](@ref) forms ``\beta = (-\partial_2 \phi, \partial_1 \phi)``, which names
both axes and exists only here.

Everything such a bracket asks of the space is the [`DiscreteSpace`](@ref) interface —
`nbasis`, `quadrature_nodes` as a vector of pairs, `quadrature_weights` as a flat vector,
`basis_values(s, d::NTuple{2,Int})` and `field` — so a third planar space joins by being added
to this union and needs no other change.
"""
const PlanarSplineSpace{T} = Union{TensorSplineSpace{T, 2}, PolarSplineSpace{T}}
