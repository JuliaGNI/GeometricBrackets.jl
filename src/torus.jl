
@doc raw"""
    TorusGrid{T, DT}

A uniform periodic grid on the two-torus ``\Omega = [0, 2\pi]^2``, carrying the
differentiation matrix of one particular scheme.

Fields are sampled as `N`-by-`N` arrays whose first index runs over ``x`` and second over
``y``, so that `f[i,j]` is the value at `(nodes[i], nodes[j])`. Everything is periodic, which
is what makes all boundary terms arising from integration by parts vanish — the assumption
under which the identities of *Poisson brackets from four-brackets* are stated.

# One type, two schemes

The two schemes differ only in the matrix `D`, so they share a type rather than forming a
hierarchy:

  * [`spectral_grid`](@ref) — Fourier differentiation, exact to roundoff for fields that are
    band-limited below the Nyquist mode. Used for the identities that are pure algebra plus
    integration by parts, where the residual is expected at `1e-15`.
  * [`finite_difference_grid`](@ref) — 8th-order centred differences, for identities
    involving non-band-limited functions of ``u`` (``u^{3/2}``, ``\log u``, quotients
    ``A_u/S_u``). Those are certified by their observed convergence rate rather than by an
    absolute threshold.

# Why a matrix and not a closure

A derivative is one matrix product: ``\partial_x f = D f`` and ``\partial_y f = f D^\top``,
both ``O(N^3)`` through BLAS for the dense spectral `D` and ``O(N^2)`` for the sparse
finite-difference one. Storing the scheme as a `Function` field instead would leave the type
abstract at every call site, and a spectral derivative evaluated by a naive DFT per call
costs ``O(N^4)`` — which is why such an implementation can only be used at very small `N`.

`D` is antisymmetric for both schemes, exactly and not just to roundoff in the
finite-difference case. That is a property of a centred stencil on a periodic grid, and
[`canonical_bracket`](@ref) inherits its antisymmetry from it.
"""
struct TorusGrid{T, DT <: AbstractMatrix{T}}
    N::Int
    h::T
    nodes::Vector{T}
    D::DT
    scheme::Symbol
end

Base.eltype(::TorusGrid{T}) where {T} = T
Base.size(g::TorusGrid) = (g.N, g.N)
Base.length(g::TorusGrid) = g.N^2

function Base.show(io::IO, g::TorusGrid)
    print(io, "TorusGrid(", g.scheme, ", N=", g.N, ")")
end

"The grid nodes in one direction; the same vector serves both ``x`` and ``y``."
nodes(g::TorusGrid) = g.nodes

@doc raw"""
    spectral_grid(N)

A [`TorusGrid`](@ref) of `N` points per direction differentiating by the discrete Fourier
transform.

The differentiation matrix is assembled once as ``D = \Re\,(V \Lambda W)``, where `W` and `V`
are the forward and inverse one-dimensional DFT matrices and ``\Lambda = \mathrm{diag}(i
k_p)`` holds the wavenumbers. For a real field this is identical to transforming, multiplying
by ``ik``, and transforming back, but it costs one matrix product per derivative instead of
two transforms.

# The wavenumber wrap, and the Nyquist mode

Wavenumbers are wrapped as `k = p <= N÷2 ? p : p - N`, so for even `N` the mode `p = N÷2`
is assigned `+N/2` rather than `-N/2`. That asymmetry is harmless: the Fourier coefficient of
a real field at the Nyquist mode is real, so its contribution to the derivative is purely
imaginary and is removed by taking the real part. Equivalently, `D` has no Nyquist component
at all, which is what leaves it antisymmetric.

Derivatives are then exact to roundoff for any field band-limited strictly below `N÷2`, and
`N = 16` is ample for the trigonometric test fields of the manuscript, whose highest mode is
three.
"""
function spectral_grid(N::Int)
    N >= 2 ||
        throw(ArgumentError("a spectral grid needs at least two points, got N = $(N)"))
    h = 2π / N
    x = [(i - 1) * h for i in 1:N]
    k = [p <= N ÷ 2 ? p : p - N for p in 0:(N - 1)]
    W = [cis(-p * x[i]) / N for p in 0:(N - 1), i in 1:N]
    V = [cis(p * x[i]) for i in 1:N, p in 0:(N - 1)]
    D = real.(V * Diagonal(im .* k) * W)
    return TorusGrid(N, h, x, D, :spectral)
end

"""
The coefficients of the 8th-order centred first-derivative stencil, for offsets one to four:
`(∂f/∂x)_i ≈ Σ_k c_k (f_{i+k} - f_{i-k}) / h`.
"""
const FD8_COEFFICIENTS = (4 / 5, -1 / 5, 4 / 105, -1 / 280)

@doc raw"""
    finite_difference_grid(N)

A [`TorusGrid`](@ref) of `N` points per direction differentiating by the 8th-order centred
stencil [`FD8_COEFFICIENTS`](@ref), stored as a sparse circulant matrix.

Used for every identity involving a function of ``u`` that is not a trigonometric polynomial
— ``u^{3/2}``, ``\log u``, the quotient ``A_u/S_u`` — where Fourier differentiation is no
longer exact and the residual is discretisation error rather than a defect of the identity.
Such a claim is settled by refinement: doubling `N` must divide the residual by a factor
approaching ``2^8 = 256``. See `scripts/verify_fourbracket_convergence.jl`, which measures
observed orders of 7.2 to 8.1.

`N` must be at least nine, so that the four-point half-stencil does not reach around the grid
and overlap itself.
"""
function finite_difference_grid(N::Int)
    nc = length(FD8_COEFFICIENTS)
    N >= 2nc + 1 || throw(ArgumentError(
        "the 8th-order stencil reaches $(nc) points either side and needs N >= $(2nc + 1), got N = $(N)"))
    h = 2π / N
    x = [(i - 1) * h for i in 1:N]
    wrap(i) = mod(i - 1, N) + 1
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for i in 1:N, k in 1:nc

        c = FD8_COEFFICIENTS[k] / h
        push!(rows, i, i)
        push!(cols, wrap(i + k), wrap(i - k))
        push!(vals, c, -c)
    end
    D = sparse(rows, cols, vals, N, N)
    return TorusGrid(N, h, x, D, :finitedifference)
end

"""
    ∂x(g, f)
    ∂y(g, f)

The partial derivatives of the sampled field `f` on the grid `g`.

`f` is indexed `f[i,j]` with `i` running over ``x``, so these are `g.D * f` and
`f * transpose(g.D)` respectively.
"""
∂x(g::TorusGrid, f::AbstractMatrix) = g.D * f

@doc (@doc ∂x)
∂y(g::TorusGrid, f::AbstractMatrix) = f * transpose(g.D)

@doc raw"""
    pinv(g::TorusGrid; rtol = sqrt(eps(T)))

The Moore–Penrose pseudoinverse ``D^+`` of the differentiation matrix, as a dense matrix. It
undoes [`∂x`](@ref) and [`∂y`](@ref) the only way a periodic derivative can be undone:
``D^+ f`` is the zero-mean field whose ``x``-derivative is `f`, and `f * transpose(pinv(g))`
does the same in ``y``.

# Why a pseudoinverse and not an inverse

``D`` is singular. A constant has zero derivative on a periodic domain, so ``\mathbf{1}`` is
in the null space of both schemes, and at even `N` the Nyquist mode is too — for
[`spectral_grid`](@ref) because its `D` carries no Nyquist component by construction, and for
[`finite_difference_grid`](@ref) because every centred stencil has symbol
``2i \sum_k c_k \sin(k\theta) / h``, which vanishes at ``\theta = \pi``. So `D` has rank
``N - 2`` at even `N`, and `D * pinv(g)` is the projector onto the complement of those two
modes rather than the identity.

# How it is built

Both `D` are circulant, so ``D = V \Lambda W`` in the DFT matrices [`spectral_grid`](@ref)
already assembles, and the pseudoinverse inverts the eigenvalues it can:
``D^+ = V \Lambda^+ W`` with ``\lambda^+ = \lambda^{-1}`` where ``|\lambda| > `` `rtol` times
the largest, and zero elsewhere. Only the first column of `D` is needed to get ``\Lambda``,
which makes this ``O(N^2)`` rather than the ``O(N^3)`` of a general `pinv`.

The result is dense even for the sparse finite-difference `D`. That is not a loss: recovering
a field from its derivative is a global operation, and no banded matrix performs it.
"""
function pinv(g::TorusGrid{T}; rtol = sqrt(eps(T))) where {T}
    c = Vector{T}(g.D[:, 1])
    W = [cis(-p * g.nodes[i]) / g.N for p in 0:(g.N - 1), i in 1:g.N]
    V = [cis(p * g.nodes[i]) for i in 1:g.N, p in 0:(g.N - 1)]
    λ = g.N .* (W * c)
    tol = rtol * maximum(abs, λ)
    λ⁺ = [abs(l) > tol ? inv(l) : zero(l) for l in λ]
    c⁺ = real.(V * λ⁺ ./ g.N)
    return [c⁺[mod(i - j, g.N) + 1] for i in 1:g.N, j in 1:g.N]
end

"""
    sample(g, fun)

Evaluate `fun(x, y)` at every node of `g`, returning the `N`-by-`N` array of values.
"""
sample(g::TorusGrid, fun) = [fun(g.nodes[i], g.nodes[j]) for i in 1:g.N, j in 1:g.N]

@doc raw"""
    integrate(g, f)

The integral of the sampled field `f` over ``\Omega = [0,2\pi]^2``, as `sum(f) * g.h^2`.

By periodicity the rectangle rule *is* the trapezoidal rule — there is no boundary node to
halve — and for a band-limited field it is exact, since every non-constant Fourier mode sums
to zero over a full period. So on a [`spectral_grid`](@ref) this quadrature contributes no
error of its own to the identities, and any residual that remains is the identity's.
"""
integrate(g::TorusGrid, f::AbstractMatrix) = sum(f) * g.h^2

@doc raw"""
    canonical_bracket(g, f, k)

The canonical bracket on the torus,

```math
[f, k] = f_x k_y - f_y k_x ,
```

equation (4.9) of *Poisson brackets from four-brackets*.

This is the Poisson bracket of the plane read as a symplectic manifold with ``x`` and ``y``
conjugate, and it is the object every bracket in [`gardner_2bracket`](@ref) and its relatives
reduces to. It is antisymmetric pointwise, and by periodicity ``\int f[k, l]`` is fully
cyclic in its three arguments.
"""
canonical_bracket(g::TorusGrid, f, k) = ∂x(g, f) .* ∂y(g, k) .- ∂y(g, f) .* ∂x(g, k)
