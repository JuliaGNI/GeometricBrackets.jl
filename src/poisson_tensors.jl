### Poisson Tensor

@doc raw"""
    PoissonTensor(tensor::AbstractArray{DT, 3})
    PoissonTensor(DT, nx, nv, f)

A bracket as a rank-3 tensor, the `N × N × N` array ``T`` of

```math
[g, h]_I = \sum_{J, K} T_{IJK} \, g_J \, h_K ,
```

with the interface of a [`DiscreteBracket`](@ref) — [`poisson_matrix`](@ref),
[`poisson_apply`](@ref) and [`poisson_derivative`](@ref) — and no index arithmetic of its own:
the tensor it wraps carries the element type and, when it is a `GridTensor`, the grid and stencil.

`PoissonTensor(T)` wraps any rank-3 array whose three extents agree; an array whose extents differ
throws a `DimensionMismatch`. `PoissonTensor(DT, nx, nv, f)` builds the tensor of a bracket
`f(I, J, K)` on a doubly periodic `nx × nv` grid, `N = nx * nv`, as a `GridTensor` of width 1 over
that grid: `f(I, J, K)` returns ``T_{IJK}`` for three `CartesianIndex`es, as an [`Arakawa`](@ref)
does, and `DT` is the element type of the entries. `f` must vanish unless `J` and `K` both lie in
the nearest-neighbour stencil of `I`: where the sums run over that stencil, they drop every
coefficient outside it, while `Array(pt)`, `pt[i, j, k]` and [`poisson_derivative`](@ref) keep it.
An `Arakawa` on a grid other than `nx × nv` throws a `DimensionMismatch`, and an `Arakawa` whose
element type is not `DT` throws an `ArgumentError`.

A tensor built on a grid takes three `CartesianIndex`es or three linear indices in the order of
`LinearIndices((nx, nv))`. The wrapper assumes axes `1:N`; offset axes are neither checked nor
claimed. An index off the grid throws a `BoundsError` and `Array(pt)` materialises the tensor.
The sums run over the stencil of every node where the tensor is a `GridTensor` whose stencil
names no node twice, and over the whole axis otherwise. Matrix-free application of a
nearest-neighbour bracket is linear in the number of nodes.

The entries have the element type `DT`. [`poisson_matrix`](@ref) promotes `DT` with the state's
element type; [`poisson_apply`](@ref) also includes the element type of the vector it multiplies.
"""
struct PoissonTensor{DT, TT <: AbstractArray{DT, 3}} <: DiscreteBracket{DT}
    tensor::TT
    stencil::Bool

    function PoissonTensor(tensor::AbstractArray{DT, 3}) where {DT}
        size(tensor, 1) == size(tensor, 2) == size(tensor, 3) || throw(DimensionMismatch(
            "a PoissonTensor is an N × N × N tensor, but the extents of this one are " *
            "$(size(tensor))"))
        new{DT, typeof(tensor)}(tensor, _stencil(tensor))
    end
end

# The sums may be restricted to the stencil of a node where the stencil is the same at every node,
# which a periodic uniform stencil is; a stencil that names a node twice, as a direction shorter
# than 2w + 1 gives, is not, and a tensor that is not a `GridTensor` has none at all.
_stencil(::AbstractArray) = false
_stencil(gt::GridTensor) = allunique(stencil_indices(gt, 1))

# The coefficient function takes Cartesian indices on the periodic grid. Explicit range axes
# preserve the two-dimensional grid type without the integer-size axis constructor's vector.
# `MultiIndexLazyArray` takes a function, so the closure accepts callable structs too.
function PoissonTensor(DT, nx, nv, f)
    _check_bracket(f, DT, nx, nv)
    nx, nv = Int(nx), Int(nv)
    ax = MultiIndexAxis(1:nx, 1:nv)
    return PoissonTensor(GridTensor(
        MultiIndexLazyArray(DT, (I, J, K) -> f(I, J, K), ax, ax, ax), 1))
end

# a bracket that knows its grid and its element type, as an `Arakawa` does, adds a method that
# checks them against those of the tensor; any other bracket is not checked
_check_bracket(f, DT, nx, nv) = nothing

Base.size(pt::PoissonTensor) = size(pt.tensor)
Base.size(pt::PoissonTensor, i) = size(pt.tensor, i)

Base.getindex(pt::PoissonTensor, inds...) = getindex(pt.tensor, inds...)
Base.Array(pt::PoissonTensor) = Array(pt.tensor)

# The indices a sum at the node `I` runs over: the stencil of a tensor whose sums may be
# restricted to it, and the whole axis otherwise. Every caller hands the result to a helper
# below, which is what keeps each loop on one concrete iterator and so allocation-free.
function _indices(pt::PoissonTensor, I::Int)
    pt.stencil ? stencil_indices(pt.tensor, I) : Base.OneTo(size(pt.tensor, 1))
end

function _check_state(pt::PoissonTensor, û::AbstractVector)
    N = size(pt, 1)
    length(û) == N || throw(DimensionMismatch(
        "a PoissonTensor of $N nodes in each direction needs a vector of length $N, got " *
        "$(length(û))"))
    Base.require_one_based_indexing(û)
    return nothing
end

function _accumulate!(P::AbstractMatrix, pt::PoissonTensor, û::AbstractVector, I::Int, inds)
    ûI = û[I]
    for J in inds, K in inds

        P[J, K] += ûI * pt.tensor[I, J, K]
    end
    return nothing
end

function _accumulate!(Pc::AbstractVector, pt::PoissonTensor, û::AbstractVector,
        c::AbstractVector, I::Int, inds)
    ûI = û[I]
    for K in inds
        cK = c[K]
        for J in inds
            Pc[J] += ûI * pt.tensor[I, J, K] * cK
        end
    end
    return nothing
end

function poisson_matrix(pt::PoissonTensor{DT}, û::AbstractVector) where {DT}
    _check_state(pt, û)
    N = size(pt, 1)
    P = zeros(promote_type(DT, eltype(û)), N, N)
    for I in 1:N
        _accumulate!(P, pt, û, I, _indices(pt, I))
    end
    return P
end

function poisson_apply(pt::PoissonTensor{DT}, û::AbstractVector, c::AbstractVector) where {DT}
    _check_state(pt, û)
    _check_state(pt, c)
    N = size(pt, 1)
    Pc = zeros(promote_type(DT, eltype(û), eltype(c)), N)
    for I in 1:N
        _accumulate!(Pc, pt, û, c, I, _indices(pt, I))
    end
    return Pc
end

function poisson_derivative(pt::PoissonTensor, û::AbstractVector)
    _check_state(pt, û)
    return Array(pt.tensor)
end

### Poisson Operator

@doc raw"""
    PoissonOperator(tensor::PoissonTensor, h)

The operator ``g \mapsto [g, h]`` for a fixed `h`, as the lazy `N × N` matrix
``\sum_K T_{IJK} h_K`` of the [`PoissonTensor`](@ref) ``T``. It gives the bracket at the grid
nodes, with no quadrature weight. The sum runs over the stencil of each node where the tensor's
sums may be restricted to it, which holds every nonzero coefficient of a nearest-neighbour bracket
such as [`Arakawa`](@ref), and over the whole axis otherwise. `Matrix(po)` materialises it.

`h` holds one value per node of the tensor. Any other length throws a `DimensionMismatch`. The
element type of the operator promotes the tensor's with that of `h`.
"""
struct PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}
    tensor::PT
    hamiltonian::HT

    function PoissonOperator(tensor::PoissonTensor{DT}, h::HT) where {DT, HT}
        N = size(tensor, 1)
        length(h) == N || throw(DimensionMismatch(
            "a PoissonOperator of a tensor of $N nodes in each direction needs a Hamiltonian " *
            "of length $N, got $(length(h))"))
        new{promote_type(DT, eltype(h)), typeof(tensor), HT}(tensor, h)
    end
end

Base.size(po::PoissonOperator) = (size(po.tensor, 1), size(po.tensor, 2))
Base.size(po::PoissonOperator, i) = size(po)[i]

# The extents of the grid of the `GridTensor` a `PoissonTensor` wraps. Only a tensor on a
# two-dimensional grid has them: a dense array, and a grid of another dimension, raise a
# `MethodError` here rather than answer with the number of nodes in one direction.
_grid(pt::PoissonTensor{<:Any, <:GridTensor}) = _grid(axes(parent(pt.tensor), 1))
_grid(ax::MultiIndexAxis{2}) = size(ax)

_nx(pt::PoissonTensor) = _grid(pt)[1]
_nv(pt::PoissonTensor) = _grid(pt)[2]

_nx(t::PoissonOperator) = _nx(t.tensor)
_nv(t::PoissonOperator) = _nv(t.tensor)

function _entry(
        ::Type{DT}, pt::PoissonTensor, h::AbstractVector, i::Int, j::Int, inds) where {DT}
    x = zero(DT)
    for K in inds
        x += pt.tensor[i, j, K] * h[K]
    end
    return x
end

function Base.getindex(po::PoissonOperator{DT}, i::Int, j::Int) where {DT}
    checkbounds(po, i, j)
    return _entry(DT, po.tensor, po.hamiltonian, i, j, _indices(po.tensor, i))
end
