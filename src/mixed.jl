# The mixed two-field formulation of the implicit methods.
#
# The dense formulation solves N equations in the state alone, and its Newton matrix is dense
# because `poisson_matrix` is `Minv * K(u) * Minv` with a dense inverse mass matrix -- see
# `kernel_operator`. Writing the equations of motion as staged solves instead,
#
#     M v      = ∂H/∂u                    (v is the "momentum" M⁻¹ ∂H/∂u)
#     M du/dt  = K(u) v
#
# never forms that sandwich. Eliminating the intermediate leaves 2N equations in (y, v) whose
# Jacobian is built from M, K, ∂(Kv)/∂u and ∂²H -- all banded with circular bandwidth p, and
# all with a sparsity pattern fixed for the whole run, which is what lets the linear solver
# reuse one ordering and symbolic factorization.
#
# It is not always the better trade: 2N unknowns against N, so the dense form wins while N is
# small. Measured crossover is around N = 130; `formulation = :dense` remains the default.

"""
    mixed_dimension(method, N)

The number of unknowns the mixed formulation of `method` uses, for a state of length `N`.

`2N` for [`ImplicitMidpoint`](@ref): the state and one auxiliary. The other implicit methods
are not supported — [`AverageVectorField`](@ref) needs one auxiliary per quadrature node and
the discrete-gradient methods carry a rank-one term in `∂ḡ/∂y` that would have to be bordered
— and say so rather than silently producing a wrong system.
"""
mixed_dimension(::ImplicitMidpoint, N::Integer) = 2N

function mixed_dimension(m::IntegratorMethod, ::Integer)
    throw(ArgumentError(
        "formulation = :mixed is implemented for ImplicitMidpoint only, but got " *
        "$(nameof(typeof(m))); use formulation = :dense"))
end

"""
    mixed_state(z, N)

Views of the state `y` and the auxiliary `v` in the stacked unknown `z`.
"""
mixed_state(z::AbstractVector, N::Integer) = (view(z, 1:N), view(z, (N + 1):2N))

"""
    _pattern_union(mats...)

A sparse matrix whose stored positions are the union of those of `mats`, with unit values.

Used to build the Jacobian skeleton. The blocks' own patterns vary with the point they are
evaluated at — `kernel_directional(b, v)` has no stored entries at all for `v = 0`, since its
values come from a product with `v` — so the skeleton has to be their *structural* union,
taken once, and the per-iteration blocks scattered into it. See [`_scatter_add!`](@ref).
"""
function _pattern_union(mats::AbstractMatrix...)
    n, m = size(first(mats))
    # The blocks' own element type, not `Float64`: this matrix becomes the Jacobian prototype,
    # which is what sizes the solver's cache and its `LinearProblem`, so an `ones(...)` here
    # would hand a `Float64` factorisation to a `BigFloat` residual.
    T = promote_type(map(eltype, mats)...)
    Is, Js = Int[], Int[]
    for A in mats
        i, j, _ = findnz(sparse(A))
        append!(Is, i)
        append!(Js, j)
    end
    sparse(Is, Js, ones(T, length(Is)), n, m, (a, b) -> one(a))
end

"""
    _scatter_add!(J, A, roff, coff)

Add `A` into `J` at the block offset `(roff, coff)`, in place, without changing `J`'s pattern.

Every stored position of `A` must already be stored in `J`; `setindex!` on a `SparseMatrixCSC`
would otherwise *insert* it, reallocating and silently changing the pattern the linear solver's
symbolic factorization was built for. That is what [`mixed_jacobian_prototype`](@ref)
guarantees, and what the `nnz` check in [`mixed_jacobian!`](@ref) verifies.
"""
function _scatter_add!(J::SparseMatrixCSC, A::SparseMatrixCSC, roff::Integer, coff::Integer)
    rows, vals = rowvals(A), nonzeros(A)
    for j in axes(A, 2), k in nzrange(A, j)

        J[rows[k] + roff, j + coff] += vals[k]
    end
    J
end

function _scatter_add!(J::SparseMatrixCSC, A::AbstractMatrix, roff::Integer, coff::Integer)
    _scatter_add!(J, sparse(A), roff, coff)
end

"""
    mixed_jacobian_prototype(f, method)

The sparsity pattern of the mixed Jacobian, as a matrix of ones.

Built once and handed to the solver as its `jacobian_prototype`, so that the ordering and
symbolic factorization are computed once for the whole run. The blocks are evaluated at
generic data — `ones` and a pseudo-random vector — because a block's stored positions depend on
the point, and the skeleton must be a superset of every one of them.
"""
function mixed_jacobian_prototype(f::HamiltonianFlow{T}, method::ImplicitMidpoint) where {T}
    s = f.space
    N = nbasis(s)
    b = f.bracket
    M = sparse(mass_matrix(s))

    # Generic points: `ones` realizes the full structure of the products, and a second,
    # sign-varying sample guards against a cancellation that `ones` happens to miss.
    u₁ = ones(T, N)
    u₂ = T[cospi(2 * (i - 1) / N) for i in 1:N]

    K = _pattern_union(kernel_operator(b, u₁), kernel_operator(b, u₂))
    D = _pattern_union(kernel_directional(b, u₁), kernel_directional(b, u₂))
    H = _pattern_union(sparse(hessian(f, u₁)), sparse(hessian(f, u₂)))

    A11 = _pattern_union(M, D)
    A12 = K
    A21 = H
    A22 = M

    J = [A11 A12
         A21 A22]
    dropzeros!(J)
    fill!(nonzeros(J), one(T))
    J
end

"""
    mixed_row_scale(space)

The factor the first residual block is multiplied by, `1 / ‖M‖∞`.

Without it the two blocks of the mixed residual have incommensurable scales. `M (y - uⁿ)`
carries a factor of the mass matrix, whose row sums are the cell width `h = L/N`, while
`M v - ∂H/∂u` has the scale of the gradient and does not shrink with `N` at all. A single
absolute `f_abstol` then cannot serve both: calibrating it for the first block makes the second
unreachable, which at `N = 1024` showed up as the solver spinning to its iteration cap on a
step it had in fact solved.

Scaling the *equation* rather than the tolerance fixes this, and it is applied **uniformly** to
both row blocks. That distinction matters: scaling the blocks by different factors changes the
conditioning, and scaling only the first makes `cond(J)` 200 to 1700 times worse. A uniform
factor multiplies the whole system, so the condition number and the Newton step are both
exactly unchanged, and only the residual's magnitude moves — which is all the convergence
test needs.
"""
mixed_row_scale(s::DiscreteSpace) = inv(opnorm(Matrix(mass_matrix(s)), Inf))

"""
    mixed_residual!(r, method, f, un, z, Δt, σ)

The residual of the mixed formulation of `method` at the stacked unknown `z = [y; v]`.

For [`ImplicitMidpoint`](@ref), with `ū = (uⁿ + y)/2` and `σ` from
[`mixed_row_scale`](@ref):

```
r₁ = σ (M (y - uⁿ) - Δt K(ū) v)
r₂ = σ (M v - ∂H/∂u(ū))
```

Eliminating `v` recovers `σ M (y - uⁿ - Δt M⁻¹ K(ū) M⁻¹ ∂H/∂u(ū))`, i.e. the dense
`residual!` up to the scaling — the two describe the same step, and the tests hold them
to that.
"""
function mixed_residual!(
        r::AbstractVector, ::ImplicitMidpoint, f::HamiltonianFlow, un, z, Δt, σ)
    N = length(un)
    y, v = mixed_state(z, N)
    r₁, r₂ = mixed_state(r, N)
    ū = (un .+ y) ./ 2
    M = mass_matrix(f.space)
    one_ = one(eltype(r))

    mul!(r₁, M, y)
    mul!(r₁, M, un, -one_, one_)
    mul!(r₁, kernel_operator(f.bracket, ū), v, -Δt, one_)
    r₁ .*= σ

    mul!(r₂, M, v)
    r₂ .-= gradient(f, ū)
    r₂ .*= σ
    r
end

"""
    mixed_jacobian!(j, method, f, un, z, Δt, σ)

The exact Jacobian of [`mixed_residual!`](@ref), assembled into `j`'s existing sparsity
pattern.

For [`ImplicitMidpoint`](@ref):

```
        ∂/∂y                                    ∂/∂v
r₁   [ σ (M - (Δt/2) ∂(K(ū)v)/∂ū)         |  -σ Δt K(ū)  ]
r₂   [ -σ (1/2) ∂²H(ū)                    |   σ M        ]
```

`σ` multiplies the whole system, residual and Jacobian alike, so both the Newton step and the
condition number are unchanged by it; see [`mixed_row_scale`](@ref).

`∂(K(ū)v)/∂ū` is [`kernel_directional`](@ref), which the bracket already provides and which is
independent of `ū` because the block is linear in it. Every entry is banded, so this is where
the mixed formulation earns its keep: the dense `residual_jacobian!` forms
`Minv * K * Minv * ∂²H`, two dense `N × N` products per assembly.
"""
function mixed_jacobian!(
        j::SparseMatrixCSC, ::ImplicitMidpoint, f::HamiltonianFlow, un, z, Δt, σ)
    N = length(un)
    y, v = mixed_state(z, N)
    ū = (un .+ y) ./ 2
    b = f.bracket
    M = sparse(mass_matrix(f.space))
    nz₀ = nnz(j)

    fill!(nonzeros(j), zero(eltype(j)))
    _scatter_add!(j, σ .* M, 0, 0)
    _scatter_add!(j, (-σ * Δt / 2) .* kernel_directional(b, v), 0, 0)
    _scatter_add!(j, (-σ * Δt) .* kernel_operator(b, ū), 0, N)
    _scatter_add!(j, (-σ / 2) .* sparse(hessian(f, ū)), N, 0)
    _scatter_add!(j, σ .* M, N, N)

    # A pattern that grew means the prototype was not a superset after all, and the linear
    # solver's symbolic factorization is now stale. Cheap next to the assembly, and the
    # failure it catches is otherwise a wrong answer rather than an error.
    nnz(j) == nz₀ ||
        error("the mixed Jacobian's sparsity pattern grew during assembly, from " *
              "$(nz₀) to $(nnz(j)) stored entries; mixed_jacobian_prototype did " *
              "not cover it")
    j
end

"""
    mixed_initial_guess!(z, f, un)

Fill the stacked unknown with the natural starting point: `y = uⁿ` and the auxiliary
`v = M⁻¹ ∂H/∂u(uⁿ)`, one mass solve.
"""
function mixed_initial_guess!(z::AbstractVector, f::HamiltonianFlow, un::AbstractVector)
    N = length(un)
    y, v = mixed_state(z, N)
    y .= un
    v .= gradient(f, un)
    mass_solve!(v, mass_factorization(f.space), v)
    z
end
