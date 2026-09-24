### Poisson Tensor

@doc raw"""
    PoissonTensor(DT, nx, nv, f)

The `N × N × N` tensor ``T`` of a bracket on a doubly periodic `nx × nv` grid, `N = nx * nv`,

```math
[g, h]_I = \sum_{J, K} T_{IJK} \, g_J \, h_K .
```

`f(I, J, K)` returns ``T_{IJK}`` for three `CartesianIndex`es, as an [`Arakawa`](@ref) does, and
`DT` is its element type. The tensor is lazy. It takes three `CartesianIndex`es or three linear
indices in the order of `LinearIndices((nx, nv))`, and an index off the grid throws a
`BoundsError`. `Array(pt)` materialises it.
"""
struct PoissonTensor{DT, FT}
    nx::Int
    nv::Int
    f::FT

    function PoissonTensor(DT, nx, nv, f)
        new{DT, typeof(f)}(nx, nv, f)
    end
end

Base.size(pt::PoissonTensor) = ntuple(_ -> pt.nx * pt.nv, 3)
Base.size(pt::PoissonTensor, i) = i ≥ 1 && i ≤ 3 ? pt.nx * pt.nv : 1

function Base.getindex(pt::PoissonTensor, I::CartesianIndex, J::CartesianIndex, K::CartesianIndex)
    grid = CartesianIndices((pt.nx, pt.nv))
    I in grid && J in grid && K in grid || throw(BoundsError(pt, (I, J, K)))
    pt.f(I, J, K)
end

function Base.getindex(pt::PoissonTensor, i::Int, j::Int, k::Int)
    grid = CartesianIndices((pt.nx, pt.nv))
    all(n -> checkbounds(Bool, grid, n), (i, j, k)) || throw(BoundsError(pt, (i, j, k)))
    pt[grid[i], grid[j], grid[k]]
end

function Base.Array(pt::PoissonTensor)
    N = pt.nx * pt.nv
    [pt[i, j, k] for i in 1:N, j in 1:N, k in 1:N]
end

### Poisson Operator

@doc raw"""
    PoissonOperator(tensor::PoissonTensor, h)

The operator ``g \mapsto [g, h]`` for a fixed `h`, as the lazy `N × N` matrix
``\sum_K T_{IJK} h_K`` of the [`PoissonTensor`](@ref) ``T``. It gives the bracket at the grid
nodes, with no quadrature weight. The sum runs over the ``3 \times 3`` stencil around `I`, which
holds every nonzero coefficient of a nearest-neighbour bracket such as [`Arakawa`](@ref).
`Matrix(po)` materialises it.

`h` holds one value per grid node, in the order of `LinearIndices((nx, nv))`. Any other length
throws a `DimensionMismatch`.
"""
struct PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}
    tensor::PT
    hamiltonian::HT

    function PoissonOperator(tensor::PoissonTensor{DT}, h::HT) where {DT, HT}
        length(h) == tensor.nx * tensor.nv || throw(DimensionMismatch(
            "a PoissonOperator on a $(tensor.nx) × $(tensor.nv) grid needs a Hamiltonian of " *
            "length $(tensor.nx * tensor.nv), got $(length(h))"))
        new{DT, typeof(tensor), HT}(tensor, h)
    end
end

Base.size(po::PoissonOperator) = (size(po.tensor, 1), size(po.tensor, 2))
Base.size(po::PoissonOperator, i) = size(po)[i]

_nx(t::PoissonTensor) = t.nx
_nv(t::PoissonTensor) = t.nv

_nx(t::PoissonOperator) = _nx(t.tensor)
_nv(t::PoissonOperator) = _nv(t.tensor)

function Base.getindex(po::PoissonOperator{DT}, i::Int, j::Int) where {DT}
    checkbounds(po, i, j)
    nx, nv = _nx(po), _nv(po)
    grid = CartesianIndices((nx, nv))
    li = LinearIndices(grid)
    I, J = grid[i], grid[j]

    x = zero(DT)
    for d1 in -1:1, d2 in -1:1

        K = CartesianIndex(mod1(I[1] + d1, nx), mod1(I[2] + d2, nv))
        x += po.tensor[I, J, K] * po.hamiltonian[li[K]]
    end
    return x
end
