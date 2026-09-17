### Poisson Tensor (used with h)
# A N × N × N tensor that discretizes the weak form of the Arakawa bracket g[f,h]

struct PoissonTensor{DT, FT}
    nx::Int
    nv::Int
    f::FT

    function PoissonTensor(DT, nx, nv, f)
        new{DT, typeof(f)}(nx, nv, f)
    end
end

Base.size(pt::PoissonTensor) = tuple(pt.nx * pt.nv * ones(Int, 3)...)
Base.size(pt::PoissonTensor, i) = i ≥ 1 && i ≤ 3 ? pt.nx * pt.nv : 1

function Base.getindex(pt::PoissonTensor, I::CartesianIndex, J::CartesianIndex, K::CartesianIndex)
    @assert isvalid(I, pt.nx, pt.nv)
    @assert isvalid(J, pt.nx, pt.nv)
    @assert isvalid(K, pt.nx, pt.nv)

    pt.f(I, J, K)
end

function Base.getindex(pt::PoissonTensor, i::Int, j::Int, k::Int)
    I = multiindex(i, pt.nx, pt.nv)
    J = multiindex(j, pt.nx, pt.nv)
    K = multiindex(k, pt.nx, pt.nv)

    pt[I, J, K]
end

function Base.materialize(rt::PoissonTensor)
    [rt[i, j, k] for i in 1:size(rt, 1), j in 1:size(rt, 2), k in 1:size(rt, 3)]
end

### Poisson Operator
# weak form of the Operator f ↦ [f,h]

struct PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}
    tensor::PT
    hamiltonian::HT

    function PoissonOperator(tensor::PoissonTensor{DT}, h::HT) where {DT, HT}
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
    @assert i ≥ 1 && i ≤ size(po, 1)
    @assert j ≥ 1 && j ≤ size(po, 2)

    local x = zero(DT)

    ni = _stencil_indices(i, 1, _nx(po), _nv(po)) # neighboring indices of i
    #nj = _stencil_indices(j, 1, po.tensor.nx, po.tensor.nv) # neighboring indices of j

    @inbounds for k in ni
        x += po.tensor[i, j, k] * po.hamiltonian[k]
    end

    return x
end

function Base.materialize(rt::PoissonOperator)
    [rt[i, j] for i in 1:size(rt, 1), j in 1:size(rt, 2)]
end
