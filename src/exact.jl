
@doc raw"""
    rref(A) -> (R, pivots)

Reduced row echelon form of `A` by Gauss-Jordan elimination, together with the list of
pivot columns.

The elimination is written out rather than delegated because it has to run over
``\mathbb{Q}``. `det`, `inv`, `lu` and `\` all already work on `Rational{BigInt}` through
Julia's generic fallbacks, but `rank` and `nullspace` route through the SVD and do not —
and it is precisely rank and kernel that the structural questions here ask. Over the
rationals the arithmetic is exact, so the pivot test is `iszero` and there is no tolerance
to choose.

The same code runs on floating point, where `iszero` becomes a genuinely unsafe pivot test.
That is deliberate: the exact and floating-point paths through the Dirac machinery differ
only in their element type, and keeping one implementation is what stops them drifting
apart. Pass an exact type when the answer is meant to be exact.

See also [`kernel`](@ref) and [`exact_rank`](@ref).
"""
function rref(A::AbstractMatrix{T}) where {T}
    R = Matrix{T}(copy(A))
    n, m = size(R)
    pivots = Int[]
    r = 1
    for c in 1:m
        r > n && break
        p = findnext(i -> !iszero(R[i, c]), r:n, 1)
        p === nothing && continue
        p = r + p - 1
        r != p && for j in 1:m
            R[r, j], R[p, j] = R[p, j], R[r, j]
        end
        pv = R[r, c]
        for j in 1:m
            R[r, j] /= pv
        end
        for i in 1:n
            (i == r || iszero(R[i, c])) && continue
            f = R[i, c]
            for j in 1:m
                R[i, j] -= f * R[r, j]
            end
        end
        push!(pivots, c)
        r += 1
    end
    return R, pivots
end

"""
    exact_rank(A)

The rank of `A` by exact elimination, as the number of pivots of [`rref`](@ref).

Named apart from `LinearAlgebra.rank` rather than added to it: `rank` is a foreign function
and `Matrix{Rational{BigInt}}` a foreign type, so a method on the pair would be piracy, and
the two answer different questions anyway — one counts singular values above a tolerance,
this one counts pivots exactly.
"""
exact_rank(A::AbstractMatrix) = length(last(rref(A)))

@doc raw"""
    kernel(A) -> Matrix

A basis for the null space of `A`, as the columns of the returned matrix.

Set one free variable to one and the rest to zero, then read the pivot entries off the
reduced row echelon form. Exact when the element type is, so `kernel` of an integer or
rational matrix answers "is there a Casimir here" with a yes or a no rather than with a
singular value that is small.

Returns a `0`-column matrix when `A` has trivial kernel.

# Examples

```jldoctest
julia> using GeometricBrackets

julia> kernel(Rational{Int}[1 2 3; 2 4 6])
3×2 Matrix{Rational{Int64}}:
 -2  -3
  1   0
  0   1
```
"""
function kernel(A::AbstractMatrix{T}) where {T}
    R, pivots = rref(A)
    m = size(A, 2)
    free = setdiff(1:m, pivots)
    K = zeros(T, m, length(free))
    for (col, fc) in enumerate(free)
        K[fc, col] = one(T)
        for (row, pc) in enumerate(pivots)
            K[pc, col] = -R[row, fc]
        end
    end
    return K
end
