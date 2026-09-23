### Arakawa ###

@doc raw"""
    Arakawa(nx, nv, hx, hv)

Arakawa's discretisation of the canonical bracket ``[f, h] = \partial_x f \, \partial_v h -
\partial_v f \, \partial_x h`` on a doubly periodic `nx × nv` grid with spacings `hx` and `hv`,
as a [`DiscreteBracket`](@ref) in Lie-Poisson form.

Called as `arakawa(I, J, K)` on three `CartesianIndex`es, it returns the stencil coefficient
``A(I, J, K)`` of the Arakawa Jacobian,

```math
[f, h]_I = \sum_{J, K} A(I, J, K) \, f_J \, h_K ,
```

which is nonzero only where `J` and `K` both lie in the ``3 \times 3`` stencil around `I`. It is
what one passes to `PoissonTensor` as its `f`.

As a bracket its state is the grid function ``\hat{f}`` itself, a vector of `nx * nv` node
values in the order of `LinearIndices((nx, nv))`, and the structure matrix is linear in it,

```math
\mathbb{P}(\hat{f})_{JK} = h_x h_v \sum_I \hat{f}_I \, A(I, J, K) .
```

[`poisson_apply`](@ref) contracts the stencil directly and builds no matrix.
[`poisson_derivative`](@ref) is the constant tensor ``h_x h_v A(l, i, j)``, whatever
``\hat{f}``.

Antisymmetry is exact. The sign tables satisfy ``A(I, J, K) = -A(I, K, J)`` in integers, and
[`poisson_matrix`](@ref) accumulates the entries ``(J, K)`` and ``(K, J)`` from the same terms in
the same order, so `P == -P'` holds to the last bit. The Jacobi identity does not hold:
[`jacobi_residual`](@ref) is of order one and does not decrease under refinement. The
coefficients ``h_x h_v A`` are the grid-independent integers of the sign tables divided by 12,
so no refinement can make them close into a Lie algebra.

The stencil needs `nx ≥ 3` and `nv ≥ 3`. On a shorter periodic direction the neighbours
`i - 1` and `i + 1` are the same node.

# The stencil

With `h` the second argument, the three second-order Jacobians whose mean is Arakawa's are

```
jpp =
    + f[i-1, j  ] * h[i,   j-1]
    - f[i-1, j  ] * h[i,   j+1]
    - f[i,   j-1] * h[i-1, j  ]
    + f[i,   j-1] * h[i+1, j  ]
    + f[i,   j+1] * h[i-1, j  ]
    - f[i,   j+1] * h[i+1, j  ]
    - f[i+1, j  ] * h[i,   j-1]
    + f[i+1, j  ] * h[i,   j+1]

jpc =
    + f[i-1, j  ] * h[i-1, j-1]
    - f[i-1, j  ] * h[i-1, j+1]
    - f[i,   j-1] * h[i-1, j-1]
    + f[i,   j-1] * h[i+1, j-1]
    + f[i,   j+1] * h[i-1, j+1]
    - f[i,   j+1] * h[i+1, j+1]
    - f[i+1, j  ] * h[i+1, j-1]
    + f[i+1, j  ] * h[i+1, j+1]

jcp =
    - f[i-1, j-1] * h[i-1, j  ]
    + f[i-1, j-1] * h[i,   j-1]
    + f[i-1, j+1] * h[i-1, j  ]
    - f[i-1, j+1] * h[i,   j+1]
    - f[i+1, j-1] * h[i,   j-1]
    + f[i+1, j-1] * h[i+1, j  ]
    + f[i+1, j+1] * h[i,   j+1]
    - f[i+1, j+1] * h[i+1, j  ]

return (jpp + jpc + jcp) * inv(hx) * inv(hv) / 12
```
"""
struct Arakawa{DT} <: DiscreteBracket{DT}
    nx::Int
    nv::Int
    hx::DT
    hv::DT
    factor::DT

    JPP::OffsetArray{Int, 4, Array{Int, 4}}
    JPC::OffsetArray{Int, 4, Array{Int, 4}}
    JCP::OffsetArray{Int, 4, Array{Int, 4}}

    function Arakawa(nx::Int, nv::Int, hx::DT, hv::DT) where {DT}
        nx ≥ 3 && nv ≥ 3 || throw(ArgumentError(
            "the Arakawa stencil needs at least 3 nodes per direction, got $nx × $nv"))

        JPP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JPC = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JCP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)

        JPP[-1, 0, 0, -1] = +1
        JPP[-1, 0, 0, +1] = -1
        JPP[0, -1, -1, 0] = -1
        JPP[0, -1, +1, 0] = +1
        JPP[0, +1, -1, 0] = +1
        JPP[0, +1, +1, 0] = -1
        JPP[+1, 0, 0, -1] = -1
        JPP[+1, 0, 0, +1] = +1

        JPC[-1, 0, -1, -1] = +1
        JPC[-1, 0, -1, +1] = -1
        JPC[0, -1, -1, -1] = -1
        JPC[0, -1, +1, -1] = +1
        JPC[0, +1, -1, +1] = +1
        JPC[0, +1, +1, +1] = -1
        JPC[+1, 0, +1, -1] = -1
        JPC[+1, 0, +1, +1] = +1

        JCP[-1, -1, -1, 0] = -1
        JCP[-1, -1, 0, -1] = +1
        JCP[-1, +1, -1, 0] = +1
        JCP[-1, +1, 0, +1] = -1
        JCP[+1, -1, 0, -1] = -1
        JCP[+1, -1, +1, 0] = +1
        JCP[+1, +1, 0, +1] = +1
        JCP[+1, +1, +1, 0] = -1

        factor = inv(hx) * inv(hv) / 12

        new{DT}(nx, nv, hx, hv, factor, JPP, JPC, JCP)
    end
end

mymod(i, n, w = 1) = abs(i) ≥ n - w ? i - n * sign(i) : i

function (arakawa::Arakawa{DT})(I, J, K) where {DT}
    fi = mymod.(Tuple(J - I), (arakawa.nx, arakawa.nv))
    hi = mymod.(Tuple(K - I), (arakawa.nx, arakawa.nv))

    if any(fi .< -1) || any(fi .> +1) || any(hi .< -1) || any(hi .> +1)
        return zero(DT)
    end

    (arakawa.JPP[fi..., hi...] +
     arakawa.JPC[fi..., hi...] +
     arakawa.JCP[fi..., hi...]) * arakawa.factor
end

### Arakawa as a DiscreteBracket ###

# A(I, J, K) vanishes unless J and K both lie in the 3 × 3 stencil around I, so every sum over
# (I, J, K) below runs over I and two offsets from it. I is the outer loop, in linear order.
const _ARAKAWA_OFFSETS = CartesianIndices((-1:1, -1:1))

_wrap(b::Arakawa, I::CartesianIndex{2}) = CartesianIndex(mod1(I[1], b.nx), mod1(I[2], b.nv))

function _check_state(b::Arakawa, û::AbstractVector)
    length(û) == b.nx * b.nv || throw(DimensionMismatch(
        "an Arakawa bracket on a $(b.nx) × $(b.nv) grid needs a vector of length " *
        "$(b.nx * b.nv), got $(length(û))"))
end

# The entries (J, K) and (K, J) receive the same terms, negated, in the same order of I, which
# is what makes the assembled matrix antisymmetric to the last bit.
function poisson_matrix(b::Arakawa{DT}, û::AbstractVector) where {DT}
    _check_state(b, û)
    li = LinearIndices((b.nx, b.nv))
    P = zeros(promote_type(DT, eltype(û)), length(li), length(li))
    for I in CartesianIndices(li)
        w = b.hx * b.hv * û[li[I]]
        for d in _ARAKAWA_OFFSETS, e in _ARAKAWA_OFFSETS

            J = _wrap(b, I + d)
            K = _wrap(b, I + e)
            P[li[J], li[K]] += w * b(I, J, K)
        end
    end
    return P
end

function poisson_apply(b::Arakawa{DT}, û::AbstractVector, c::AbstractVector) where {DT}
    _check_state(b, û)
    _check_state(b, c)
    T = promote_type(DT, eltype(û), eltype(c))
    li = LinearIndices((b.nx, b.nv))
    Pc = zeros(T, length(li))
    for I in CartesianIndices(li)
        w = b.hx * b.hv * û[li[I]]
        for d in _ARAKAWA_OFFSETS
            J = _wrap(b, I + d)
            s = zero(T)
            for e in _ARAKAWA_OFFSETS
                K = _wrap(b, I + e)
                s += b(I, J, K) * c[li[K]]
            end
            Pc[li[J]] += w * s
        end
    end
    return Pc
end

function poisson_derivative(b::Arakawa{DT}, û::AbstractVector) where {DT}
    _check_state(b, û)
    li = LinearIndices((b.nx, b.nv))
    N = length(li)
    dP = zeros(DT, N, N, N)
    for I in CartesianIndices(li), d in _ARAKAWA_OFFSETS, e in _ARAKAWA_OFFSETS
        J = _wrap(b, I + d)
        K = _wrap(b, I + e)
        dP[li[I], li[J], li[K]] = b.hx * b.hv * b(I, J, K)
    end
    return dP
end
