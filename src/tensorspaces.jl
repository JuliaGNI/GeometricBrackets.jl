
@doc raw"""
    TensorSplineSpace(quadrature::TensorProductQuadrature)
    TensorSplineSpace(basis::TensorProductBasis; kwargs...)
    TensorSplineSpace(bases::AbstractBSplineBasis...; kwargs...)
    TensorSplineSpace(meshes::NTuple{D,Mesh}, p, bc = Periodic(); kwargs...)
    TensorSplineSpace(n::NTuple{D,Integer}, p, bc = Periodic(); L = 2π, kwargs...)

The tensor-product B-spline space
``\Phi_{i_1 \dots i_D}(x) = \prod_{d} \phi^{(d)}_{i_d}(x_d)`` on the box
``\Omega = \Omega_1 \times \dots \times \Omega_D``, a thin wrapper around the assembly tables
of SimpleSplines.

```jldoctest
julia> s = TensorSplineSpace((8, 8), 3);

julia> nbasis(s), degree(s)
(64, (3, 3))
```

Degree, mesh, domain and boundary condition are **per axis**, and all three are reached
through the one dispatching constructor `BSplineBasis(mesh, p, bc)`: `Periodic()` gives the
periodic torus, `Dirichlet()` the recombined homogeneous-Dirichlet basis, `Free()` the plain
clamped one. A scalar `p`, `bc` or `L` is applied to every axis; a `D`-tuple gives one entry
per axis. To impose *different* conditions at the two ends of a single axis, build that
axis's basis yourself and pass the bases — `(Dirichlet(), Neumann())` as `bc` on a
two-dimensional space is read as one condition per axis, not as the two ends of one.

# Coefficients are flat vectors

Every bracket, Hamiltonian, flow and integrator in this package is written against
[`DiscreteSpace`](@ref) and takes a coefficient **vector** of length ``N = \prod_d N_d``.
That convention is kept here, and the array shape `size(space)` that SimpleSplines uses
appears only where it must — [`evaluate`](@ref) reshapes into it, and `KroneckerMass`
reshapes on its own. The flattening runs the **first axis fastest**, which is
`LinearIndices(basis(space))`, `vec` of a `quadrature_sample` array, and the convention that
makes the assembled mass matrix `kron(M_D, …, M_1)`. All three agree, and none of the code
below would be correct if they did not.

# What generalises, and what does not

The abstract interface is answered unchanged, so [`project`](@ref), [`project!`](@ref),
[`basis_integrals`](@ref) and `field` are the generic implementations of `spaces.jl` running
on a two- or three-dimensional space. The one notion that has no scalar meaning here is the
derivative order: ``\partial_1 \phi`` and ``\partial_2 \phi`` are different tables, so
`basis_values(space, d)` takes a **per-axis multi-index** `d::NTuple{D,Int}`, and the scalar
form is accepted only for `d = 0` and rejected with a message naming the tuple otherwise.
`derivative_matrix` and `stiffness_matrix` follow: the first takes the axis `k` it
differentiates along, and the second is ``\int_\Omega \nabla \phi_k \cdot \nabla \phi_l``,
the sum over the axes. [`domainlength`](@ref) returns the per-axis tuple rather than a
scalar; the scalar that does exist is [`domainvolume`](@ref).

# Storage

The tabulation ``\Phi_d`` is the sparse Kronecker product of the ``D`` one-dimensional
tables, formed on first use and memoised, as are the [`mixed_matrix`](@ref) constants built
from it. It is ``N \times Q`` with ``\prod_d (p_d+1)`` nonzeros per column — at ``64^2``
cubic cells that is 1.6 million entries, some 26 MB per derivative multi-index, and the
three tables a gradient assembly needs are the working set. This is the price of making the
promise of [`DiscreteSpace`](@ref) literally true in ``D`` dimensions: a matrix
``\int_\Omega f \, D^a \phi_k \, D^b \phi_l`` is still
``\Phi_a \, \mathrm{diag}(f \odot w) \, \Phi_b^T`` with a *non-separable* ``f``, which no
sequence of one-dimensional contractions gives.

The mass matrix is the exception and is never inverted densely: `mass_factorization` returns
the `KroneckerMass` of SimpleSplines, whose solve is ``D`` one-dimensional solves applied
along each axis. See [`inverse_mass_matrix`](@ref) for what the dense form costs and why it
is not stored.
"""
struct TensorSplineSpace{T, D, QT <: TensorProductQuadrature{T, D}} <: DiscreteSpace{T}
    quadrature::QT
    x::Vector{NTuple{D, T}}
    w::Vector{T}
    Φ::Dict{NTuple{D, Int}, SparseMatrixCSC{T, Int}}
    cache::Dict{Tuple{NTuple{D, Int}, NTuple{D, Int}}, SparseMatrixCSC{T, Int}}

    function TensorSplineSpace(q::TensorProductQuadrature{T, D}) where {T, D}
        # `Iterators.product` runs its first factor fastest, which is the same order as
        # `vec` of a `quadrature_sample` array and as `LinearIndices` of the basis. The
        # Kronecker products below are built in reverse axis order to match it.
        x = vec([NTuple{D, T}(pt) for pt in Iterators.product(quadrature_nodes(q)...)])
        w = vec([prod(ω) for ω in Iterators.product(quadrature_weights(q)...)])

        new{T, D, typeof(q)}(q, x, w,
            Dict{NTuple{D, Int}, SparseMatrixCSC{T, Int}}(),
            Dict{Tuple{NTuple{D, Int}, NTuple{D, Int}}, SparseMatrixCSC{T, Int}}())
    end
end

function TensorSplineSpace(B::TensorProductBasis; kwargs...)
    TensorSplineSpace(TensorProductQuadrature(B; kwargs...))
end

function TensorSplineSpace(bases::AbstractBSplineBasis...; kwargs...)
    TensorSplineSpace(TensorProductBasis(bases...); kwargs...)
end

function TensorSplineSpace(meshes::NTuple{D, Mesh}, p, bc = Periodic();
        kwargs...) where {D}
    ps = _per_axis(p, D, "the degree")
    bcs = _per_axis(bc, D, "the boundary condition")
    TensorSplineSpace(
        TensorProductBasis(ntuple(
            k -> BSplineBasis(meshes[k], ps[k], bcs[k]), D)); kwargs...)
end

function TensorSplineSpace(n::NTuple{D, Integer}, p, bc = Periodic(); L = 2π,
        kwargs...) where {D}
    Ls = _per_axis(L, D, "the domain")
    TensorSplineSpace(ntuple(k -> UniformMesh(n[k], Ls[k]), D), p, bc; kwargs...)
end

# `NTuple{0, X}` is `Tuple{}` whatever `X` is, so the two constructors above overlap there and
# dispatch on `()` is ambiguous. A space with no axes is a mistake rather than a case to
# resolve between them.
function TensorSplineSpace(::Tuple{}, p, bc = Periodic(); kwargs...)
    throw(ArgumentError("a tensor-product space needs at least one axis"))
end

# A per-axis argument given once, or once per axis. Anything that is not a `Tuple` is
# broadcast; a `Tuple` of the wrong length is a mistake rather than something to recycle.
_per_axis(v, D::Integer, what::AbstractString) = ntuple(_ -> v, D)

function _per_axis(v::Tuple, D::Integer, what::AbstractString)
    length(v) == D || throw(DimensionMismatch(
        "$(what) was given for $(length(v)) axes but the space has $(D)"))
    return v
end

# The multi-index of ∂_k, i.e. the k-th unit vector of derivative orders.
_unit_index(D::Integer, k::Integer) = ntuple(i -> i == k ? 1 : 0, D)

quadrature(s::TensorSplineSpace) = s.quadrature
basis(s::TensorSplineSpace) = basis(s.quadrature)
nbasis(s::TensorSplineSpace) = nbasis(s.quadrature)
degree(s::TensorSplineSpace) = degree(basis(s))
order(s::TensorSplineSpace) = order(basis(s))
ncells(s::TensorSplineSpace) = ncells(basis(s))
nodes(s::TensorSplineSpace) = nodes(basis(s))
quadrature_nodes(s::TensorSplineSpace) = s.x
quadrature_weights(s::TensorSplineSpace) = s.w
mass_operator(s::TensorSplineSpace) = mass_operator(s.quadrature)
mass_factorization(s::TensorSplineSpace) = mass_operator(s.quadrature)
mass_matrix(s::TensorSplineSpace) = mass_matrix(mass_operator(s.quadrature))

Base.ndims(::TensorSplineSpace{T, D}) where {T, D} = D
Base.size(s::TensorSplineSpace) = size(basis(s))
Base.size(s::TensorSplineSpace, d::Integer) = size(basis(s), d)

"""
    domainlength(space::TensorSplineSpace)

The per-axis edge lengths ``(L_1, \\dots, L_D)``, as a tuple.

A tensor-product space has no single length, and returning one of the ``L_d`` — or their
product — under this name would be a wrong scalar rather than a missing one. The volume
``|\\Omega| = \\prod_d L_d`` is [`domainvolume`](@ref).
"""
domainlength(s::TensorSplineSpace) = map(domainlength, bases(basis(s)))

"""
    domainvolume(space)

The measure ``|\\Omega|`` of the domain: the length in one dimension, the area in two, the
volume in three.

This is what the mass matrix of a partition of unity sums to, and hence the reference the
assembly of a tensor-product space is checked against.
"""
domainvolume(s::TensorSplineSpace) = prod(domainlength(s))
domainvolume(s::DiscreteSpace) = domainlength(s)

@doc raw"""
    basis_values(space::TensorSplineSpace, d::NTuple{D,Int})
    basis_values(space::TensorSplineSpace, d::Integer = 0)

The table ``\Phi_d[I,R] = \partial_1^{d_1} \cdots \partial_D^{d_D} \Phi_I(x_R)`` over the
flattened quadrature grid, sparse, formed on first use and memoised.

`d` is a **per-axis multi-index**: `(1,0)` is ``\partial_1``, `(0,1)` is ``\partial_2``,
`(1,1)` is ``\partial_1 \partial_2``. Only the mixed derivatives the one-dimensional
quadratures tabulated exist, i.e. `d[k] ≤ dmax[k]`.

The scalar form is accepted only for `d = 0`, where the multi-index is unambiguous. A scalar
`d ≥ 1` is **rejected**: on more than one axis there is no such thing as "the `d`-th
derivative", and silently choosing an axis — or reading the scalar as a total order, which
`(2,0)` and `(1,1)` both satisfy — would answer a question that was not asked. Keeping `d = 0`
meaningful is what lets [`project`](@ref), [`basis_integrals`](@ref) and `field` stay the
generic implementations of `spaces.jl`.

The table is the Kronecker product ``\Phi^{(D)}_{d_D} \otimes \dots \otimes \Phi^{(1)}_{d_1}``
of the one-dimensional tables, in reverse axis order because the flattening runs the first
axis fastest.

!!! warning "The result is the cache, not a copy"
    Every call for the same `d` hands back the one memoised matrix, so mutating it corrupts
    every later call on that space. Take a `copy` before writing into it. The same holds for
    [`mixed_matrix`](@ref), and this is the only aliasing on either: the `D = 1` branch below
    copies the quadrature's own table, so the memo never aliases the quadrature.
"""
function basis_values(s::TensorSplineSpace{T, D}, d::NTuple{D, Int}) where {T, D}
    get!(s.Φ, d) do
        tabs = ntuple(k -> basis_values(quadratures(s.quadrature)[k], d[k]), D)
        # `foldl` over a one-element tuple hands back the quadrature's own table, which the
        # caller may mutate; on one axis the product is a copy instead.
        D == 1 ? copy(only(tabs)) : foldl(kron, reverse(tabs))
    end
end

function basis_values(s::TensorSplineSpace{T, D}, d::Integer = 0) where {T, D}
    d == 0 || throw(ArgumentError(
        "the derivative order of a $(D)-dimensional space is a per-axis multi-index, not " *
        "the scalar $(d): ∂₁ and ∂₂ are different tables. Write " *
        "basis_values(space, $(_unit_index(D, 1))) for ∂₁"))
    basis_values(s, ntuple(_ -> 0, D))
end

@doc raw"""
    mixed_matrix(space::TensorSplineSpace, a::NTuple{D,Int}, b::NTuple{D,Int})

The matrix ``\int_\Omega D^a \Phi_K \, D^b \Phi_L \, dx``, with `a` and `b` per-axis
derivative multi-indices, memoised.

These are constants of the discretisation — the mass matrix is `a = b = 0` and the
``\partial_k \partial_l`` blocks of [`tensor_weighted_matrix`](@ref) with a constant
coefficient are the rest — so they are assembled once per space rather than once per Newton
iteration, as they are for a `SplineSpace`.

Memoised means the result is the cache itself, as for [`basis_values`](@ref): mutating it
corrupts every later call on that space, so take a `copy` before writing into it.
"""
function mixed_matrix(s::TensorSplineSpace{T, D}, a::NTuple{D, Int},
        b::NTuple{D, Int}) where {T, D}
    get!(s.cache, (a, b)) do
        A = basis_values(s, a) * Diagonal(s.w) * basis_values(s, b)'
        SparseMatrixCSC{T, Int}(A)
    end
end

@doc raw"""
    weighted_matrix(space::TensorSplineSpace, f, a::NTuple{D,Int}, b::NTuple{D,Int})

The matrix ``\int_\Omega f(x) \, D^a \Phi_K \, D^b \Phi_L \, dx``, with `f` either a function
of a `D`-tuple of coordinates or a vector already sampled at `quadrature_nodes`.

Not memoised, unlike [`mixed_matrix`](@ref): the coefficient of a metric bracket depends on
the state and changes every Newton iteration.
"""
function weighted_matrix(s::TensorSplineSpace{T, D}, f, a::NTuple{D, Int},
        b::NTuple{D, Int}) where {T, D}
    weighted_matrix(s, map(f, quadrature_nodes(s)), a, b)
end

function weighted_matrix(s::TensorSplineSpace{T, D}, f::AbstractVector, a::NTuple{D, Int},
        b::NTuple{D, Int}) where {T, D}
    length(f) == length(s.w) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(f)) points but the quadrature grid has " *
        "$(length(s.w))"))
    basis_values(s, a) * Diagonal(f .* s.w) * basis_values(s, b)'
end

@doc raw"""
    derivative_matrix(space::TensorSplineSpace, k)

The matrix ``\mathbb{S}^{(k)}_{KL} = \int_\Omega \Phi_K \, \partial_k \Phi_L \, dx``,
antisymmetric on an axis that is periodic and not otherwise.

There is no axis-free `derivative_matrix` on a tensor-product space: ``\partial_1`` and
``\partial_2`` are different operators, and the scalar-index form inherited from
[`DiscreteSpace`](@ref) therefore raises rather than picking one.
"""
function derivative_matrix(s::TensorSplineSpace{T, D}, k::Integer) where {T, D}
    mixed_matrix(s, ntuple(_ -> 0, D), _unit_index(D, k))
end

@doc raw"""
    stiffness_matrix(space::TensorSplineSpace)

The matrix
``\mathbb{K}_{KL} = \int_\Omega \nabla \Phi_K \cdot \nabla \Phi_L \, dx
 = \sum_k \int_\Omega \partial_k \Phi_K \, \partial_k \Phi_L \, dx``,
symmetric positive semi-definite and the ``D``-dimensional counterpart of the one-dimensional
``\int \phi_k' \phi_l'``.

On the periodic torus its generalised eigenvalues against the mass matrix approximate the
spectrum ``|\mathbf{k}|^2`` of ``-\Delta``; on a homogeneous-Dirichlet box they approximate
the Dirichlet spectrum, and the constants are then no longer in the kernel.
"""
function stiffness_matrix(s::TensorSplineSpace{T, D}) where {T, D}
    sum(mixed_matrix(s, _unit_index(D, k), _unit_index(D, k)) for k in 1:D)
end

@doc raw"""
    tensor_weighted_matrix(space::TensorSplineSpace, 𝔻)

The matrix

```math
\mathbb{A}_{KL} = \int_\Omega \partial_k \Phi_K \, \mathbb{D}_{kl}(x) \,
                  \partial_l \Phi_L \, dx ,
```

summed over ``k, l = 1 \dots D`` — the variable-**tensor**-coefficient stiffness matrix that
a metric bracket assembles, and the one operator this space exists for. It is the sum of
``D^2`` scalar-weighted matrices, each [`weighted_matrix`](@ref) against one component of the
coefficient.

The coefficient may be given as

  - a constant `D×D` matrix of numbers, in which case the ``D^2`` constant
    [`mixed_matrix`](@ref) blocks are reused and only the nonzero components are touched;
  - a `D×D` matrix whose `[k,l]` entry is a vector of samples at `quadrature_nodes` —
    the canonical form, and the one a moment expansion produces component by component;
  - a vector of `D×D` matrices, one per quadrature point;
  - a function of a `D`-tuple of coordinates returning a `D×D` matrix.

The component form has to be built element by element. `[a b; c d]` with vector entries
*concatenates* them into one long matrix of numbers instead of a `D×D` matrix of vectors:

```julia
𝔻 = Matrix{Vector{Float64}}(undef, 2, 2)
𝔻[1,1], 𝔻[1,2], 𝔻[2,1], 𝔻[2,2] = a, b, c, d
```

``\mathbb{A}`` is symmetric exactly when ``\mathbb{D}`` is: transposing swaps ``K`` and ``L``,
which is the same as transposing ``\mathbb{D}``. It is positive semi-definite when
``\mathbb{D}`` is pointwise positive semi-definite, since
``v^T \mathbb{A} v = \int (\nabla v_h)^T \mathbb{D} \, \nabla v_h``. Neither is imposed here;
both are properties of the coefficient handed in, and are what the metric brackets are
checked against.
"""
function tensor_weighted_matrix(s::TensorSplineSpace{T, D},
        𝔻::AbstractMatrix{<:AbstractVector}) where {T, D}
    size(𝔻) == (D, D) || throw(DimensionMismatch(
        "the coefficient is $(size(𝔻)) but the space is $(D)-dimensional"))
    sum(weighted_matrix(s, 𝔻[k, l], _unit_index(D, k), _unit_index(D, l))
    for k in 1:D, l in 1:D)
end

function tensor_weighted_matrix(s::TensorSplineSpace{T, D},
        𝔻::AbstractMatrix{<:Number}) where {T, D}
    size(𝔻) == (D, D) || throw(DimensionMismatch(
        "the coefficient is $(size(𝔻)) but the space is $(D)-dimensional"))
    A = spzeros(T, nbasis(s), nbasis(s))
    for k in 1:D, l in 1:D

        iszero(𝔻[k, l]) && continue
        A += 𝔻[k, l] * mixed_matrix(s, _unit_index(D, k), _unit_index(D, l))
    end
    return A
end

function tensor_weighted_matrix(s::TensorSplineSpace{T, D},
        𝔻::AbstractVector{<:AbstractMatrix}) where {T, D}
    length(𝔻) == length(s.w) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(𝔻)) points but the quadrature grid has " *
        "$(length(s.w))"))
    tensor_weighted_matrix(s, [[𝔻[q][k, l] for q in eachindex(𝔻)] for k in 1:D, l in 1:D])
end

function tensor_weighted_matrix(s::TensorSplineSpace{T, D}, 𝔻) where {T, D}
    tensor_weighted_matrix(s, map(𝔻, quadrature_nodes(s)))
end

@doc raw"""
    inverse_mass_matrix(space::TensorSplineSpace)

The dense inverse mass matrix ``\mathbb{M}^{-1}``, formed **on every call** and not stored.

The Kronecker structure makes the arithmetic cheap — ``\mathbb{M}^{-1} =
(\mathbb{M}^{(D)})^{-1} \otimes \dots \otimes (\mathbb{M}^{(1)})^{-1}`` is an identity, so
this is ``D`` small inversions and one Kronecker product rather than an ``O(N^3)`` solve —
but the *storage* is ``N^2``, which at ``N = 67^2`` is 161 MB and grows as the fourth power
of the resolution. A `SplineSpace` forms this once in its constructor because at ``N = 384``
it is free; here it is not, and a space that built it eagerly could not be constructed at the
resolutions these runs reach.

So it exists, because the generic assemblies of `spaces.jl` name it, and it is the wrong
thing to call in a loop. `mass_factorization` returns the `KroneckerMass`, whose
solve is ``D`` one-dimensional solves along each axis; `\\`, `mass_solve!`, [`project`](@ref)
and [`project!`](@ref) all go through it and none of them form this matrix.
"""
function inverse_mass_matrix(s::TensorSplineSpace{T, D}) where {T, D}
    factors = map(mass_factors(mass_operator(s))) do op
        n = size(mass_matrix(op), 1)
        op \ Matrix{T}(I, n, n)
    end
    D == 1 ? only(factors) : foldl(kron, reverse(factors))
end

@doc raw"""
    evaluate(space::TensorSplineSpace, û, x, d = ntuple(_ -> 0, D))

The mixed derivative ``\partial_1^{d_1} \cdots \partial_D^{d_D} u_h`` of the field with
coefficient vector `û` at the point `x`, or at each point of a vector of points, neither of
which need lie on the quadrature grid.

`x` is any indexable `D`-vector — a tuple, an `SVector`, a length-`D` `Vector`. `û` is the
flat vector of length `nbasis`; it is reshaped here, and only here, into the
`size(space)` coefficient array SimpleSplines evaluates against.
"""
function evaluate(s::TensorSplineSpace{T, D}, û::AbstractVector, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    length(û) == nbasis(s) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the space has $(nbasis(s))"))
    evaluate(basis(s), reshape(û, size(s)), x, d)
end

function evaluate(s::TensorSplineSpace{T, D}, û::AbstractVector,
        X::AbstractVector{<:NTuple{D, Number}},
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    [evaluate(s, û, x, d) for x in X]
end

"""
    field(space::TensorSplineSpace, û, d::NTuple{D,Int})

The mixed derivative `d` of ``u_h = \\sum_I \\hat{u}_I \\Phi_I`` sampled on the flattened
quadrature grid — the ``D``-dimensional form of the operation every variable-coefficient
assembly starts from.
"""
function field(s::TensorSplineSpace{T, D}, û::AbstractVector,
        d::NTuple{D, Int}) where {T, D}
    basis_values(s, d)' * û
end
