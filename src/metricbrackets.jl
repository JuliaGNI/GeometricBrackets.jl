
@doc raw"""
    MetricBracket{T}

A discrete metric bracket, i.e. a matrix-valued function ``\mathbb{G}(\hat{u})`` through
which the symmetric bracket of two functions of the degrees of freedom is

```math
(A, B)_d = \sum_{i,j} \frac{\partial A}{\partial \hat{u}_i} \,
           \mathbb{G}_{ij} (\hat{u}) \, \frac{\partial B}{\partial \hat{u}_j} .
```

This is the symmetric half of a metriplectic structure and the mirror image of
[`DiscreteBracket`](@ref). The interface is [`metric_matrix`](@ref), [`metric_apply`](@ref)
and [`metric_derivative`](@ref); [`issymmetric`](@ref), [`ispositive_semidefinite`](@ref) and
the three-argument [`degeneracy_residual`](@ref) follow from those and are written once.

The *two*-argument `degeneracy_residual` is the one exception, and it needs two hooks more:
[`space`](@ref), and the internal `_generator(b, û)` below. It has to, because supplying the
gradient from the bracket means knowing where the bracket keeps it. A bracket that implements
only the three interface methods is not defective — it simply passes its gradient explicitly
and uses the three-argument form.

# What is and is not guaranteed

Symmetry and positive semi-definiteness are properties of how the bracket is *assembled*.
Both brackets here are built in explicitly symmetric form from a pointwise Gram matrix, so
both hold for any quadrature and any mesh.

The degeneracy is the one that has content. A metric bracket is required to satisfy
``(A, H)_d = 0`` for every ``A``, i.e. ``\mathbb{G} \, \partial H / \partial \hat{u} = 0``,
and it is *that* which makes ``\dot{H} = 0`` a property of the bracket rather than of the
quadrature or the mesh. [`degeneracy_residual`](@ref) measures it.

It is not a property of the *integrator*, and does not become one. Whether a one-step method
holds ``H`` exactly is a separate question about the method: the midpoint rule does, because a
quadratic ``H`` satisfies ``H(y) - H(x) = \nabla H(\bar{u}) \cdot (y - x)`` exactly and the
increment lies in the range of ``\mathbb{G}(\bar{u})``. Explicit Euler's increment lies in the
range of ``\mathbb{G}(\hat{u}^n)`` just as squarely and still moves ``H`` by
``O(\Delta t^2)``. See [`MetriplecticFlow`](@ref).

| bracket | symmetric | positive semi-definite | degenerate on ``H`` |
|:--|:--|:--|:--|
| [`DoubleBracket`](@ref) | yes, ``\mathbb{D} = X_h \otimes X_h`` is | yes, a pointwise Gram matrix | yes — ``X_h \cdot \nabla h = 0`` pointwise, so it holds at the quadrature level |
| [`ProjectorBracket`](@ref) | yes, a rank-one correction to ``\mathbb{M}^{-1}`` | yes, by Cauchy-Schwarz in the ``\mathbb{M}``-inner product | yes, exactly — ``\Pi_H \phi = 0`` by construction |

Both brackets are *generated* by the Hamiltonian: the double bracket by the vector field of
``\delta H / \delta u`` and the projector bracket by the direction it projects out. The
coefficients of ``\delta H / \delta u`` are what the internal hook `_generator(b, û)` returns,
and the two-argument [`degeneracy_residual`](@ref) uses it to check the defining property
without being told the Hamiltonian a second time.
"""
abstract type MetricBracket{T} end

Base.eltype(::MetricBracket{T}) where {T} = T

"""
    metric_matrix(bracket, û)

The dense metric matrix ``\\mathbb{G}(\\hat{u})``.
"""
function metric_matrix end

"""
    metric_apply(bracket, û, c)

The product ``\\mathbb{G}(\\hat{u}) \\, c``, formed without assembling the matrix where the
bracket admits it.
"""
function metric_apply(b::MetricBracket, û::AbstractVector, c::AbstractVector)
    metric_matrix(b, û) * c
end

@doc raw"""
    metric_derivative(bracket, û)

The derivative tensor `dG[l,i,j]` ``= \partial \mathbb{G}_{ij} / \partial \hat{u}_l``, which
is what the Jacobian of a metriplectic flow needs.

``O(N^3)`` in storage and, for [`DoubleBracket`](@ref), ``N`` assemblies to build — which is
why it is a separate entry point rather than something [`metric_matrix`](@ref) returns
alongside, exactly as for [`poisson_derivative`](@ref).
"""
function metric_derivative end

@doc raw"""
    metric_directional(bracket, û, v)

The matrix ``D_{im} = \partial \left( \mathbb{G}(\hat{u}) \, v \right)_i /
\partial \hat{u}_m`` at fixed `v`, i.e. the derivative tensor of
[`metric_derivative`](@ref) already contracted against one direction.

This is the symmetric-half counterpart of [`bracket_directional`](@ref) and exists for the
same reason. Differentiating ``\mathbb{G}(\hat{u}) \, \partial S/\partial \hat{u}`` produces
one term from the entropy and one from the bracket, and this is the second — the only part of
``\partial \mathbb{G}/\partial \hat{u}`` a Newton iteration ever needs. The full tensor is
``O(N^3)`` in storage and costs ``N`` sandwiches to build, which is not affordable inside a
time loop; contracting first replaces each sandwich by a single mass solve against the one
direction that is actually wanted.

The generic method here does form the tensor, so a bracket that defines only the three
interface methods still has a correct Jacobian. Every bracket in the package overrides it: a
prescribed generating field makes it identically zero, and the ``\Lambda``-generated forms are
closed or matrix-free.
"""
function metric_directional(b::MetricBracket{T}, û::AbstractVector,
        v::AbstractVector) where {T}
    dG = metric_derivative(b, û)
    N = length(û)
    D = zeros(T, N, N)
    @inbounds for m in 1:N, i in 1:N

        s = zero(T)
        for j in 1:N
            s += dG[m, i, j] * v[j]
        end
        D[i, m] = s
    end
    return D
end

"""
    issymmetric(bracket, û; atol = 1e-12)

Whether ``\\mathbb{G}(\\hat{u})`` is symmetric to within `atol`, relative to its own scale.
"""
function issymmetric(b::MetricBracket, û::AbstractVector; atol = 1e-12)
    G = metric_matrix(b, û)
    maximum(abs, G - G') ≤ atol * max(one(eltype(G)), maximum(abs, G))
end

@doc raw"""
    ispositive_semidefinite(bracket, û; atol = 1e-10)

Whether ``\mathbb{G}(\hat{u})`` has no eigenvalue below `-atol` times its largest, i.e.
whether the bracket dissipates in the direction it must.

The tolerance is relative to the largest eigenvalue because the quantity being tested is a
sign, and an absolute floor on a matrix that can be scaled arbitrarily would test the scaling
instead. Both brackets here are semi-definite by construction and genuinely singular — the
Hamiltonian is in the kernel — so the smallest eigenvalue is zero to round-off rather than
positive, and `isposdef` is the wrong question to ask of them.
"""
function ispositive_semidefinite(b::MetricBracket, û::AbstractVector; atol = 1e-10)
    λ = eigvals(Symmetric(Matrix(metric_matrix(b, û))))
    # a matrix with no positive eigenvalue needs no case of its own, because the tolerance is
    # *multiplied* by the scale rather than divided by it: at `maximum(λ) == 0` the test reads
    # `minimum(λ) ≥ 0`, which holds exactly when every eigenvalue is zero, and at
    # `maximum(λ) < 0` it demands a positive lower bound that nothing below it can meet.
    minimum(λ) ≥ -atol * maximum(λ)
end

@doc raw"""
    degeneracy_residual(bracket, û, g)
    degeneracy_residual(bracket, û)

The normalised violation of the degeneracy ``\mathbb{G}(\hat{u}) \, g = 0``,

```math
\frac{\max_i \left| \sum_j \mathbb{G}_{ij} g_j \right|}
     {\max_{ij} \left| \mathbb{G}_{ij} \right| \, \max_j \left| g_j \right|} ,
```

with `g` the gradient ``\partial H / \partial \hat{u}`` of the Hamiltonian the bracket is
generated by. The two-argument form supplies that gradient from the bracket itself, which is
``\mathbb{M} \hat{h}`` for the ``\hat{h}`` the bracket stores.

The normalisation is the same one [`jacobi_residual`](@ref) uses and is there for the same
reason: a raw residual can be driven to zero by any change that merely shrinks the bracket,
and the denominator here is the largest single product that can enter the sum.

This is the check that catches a sign error in ``Q_2``. Symmetry and positive
semi-definiteness survive one — ``X \otimes X`` is a Gram matrix whichever way ``X`` is
signed — and it is only the degeneracy that notices that the coefficient is no longer built
on the ``\perp`` of the gradient it is contracted against.
"""
function degeneracy_residual(b::MetricBracket, û::AbstractVector, g::AbstractVector)
    G = metric_matrix(b, û)
    res = maximum(abs, G * g)
    scale = maximum(abs, G) * maximum(abs, g)
    iszero(scale) ? zero(res) : res / scale
end

function degeneracy_residual(b::MetricBracket, û::AbstractVector)
    degeneracy_residual(b, û, _mass_apply(space(b), _generator(b, û)))
end

@doc raw"""
    _generator(bracket, û)

The coefficients ``\hat{h}`` of ``\delta H / \delta u``, the field the bracket is generated
by: the stream function of a [`DoubleBracket`](@ref) and the projected-out direction of a
[`ProjectorBracket`](@ref).

Both brackets store it either as a fixed vector or as the matrix ``\Lambda`` of the linear
map ``\hat{h} = \Lambda \hat{u}``. Those are the two cases the manuscript's two-dimensional
examples need — a prescribed ``h`` for the parallel-diffusion case, and
``\hat{\psi} = \mathbb{K}^{-1} \mathbb{M} \hat{\omega}`` for reduced Euler — and they are
also exactly the two for which [`metric_derivative`](@ref) is analytic.

The fallback below reads that field from `b.h`, which is the one assumption about layout the
three brackets here share. A bracket that keeps its generator elsewhere overrides this method
rather than renaming its field.
"""
_generator(b::MetricBracket, û::AbstractVector) = _apply_generator(b.h, û)

_apply_generator(h::AbstractVector, û::AbstractVector) = h
_apply_generator(Λ::AbstractMatrix, û::AbstractVector) = Λ * û

@doc raw"""
    _mass_sandwich(space, A)
    _mass_sandwich(factorization, A)

The dense ``\mathbb{M}^{-1} \mathbb{A} \mathbb{M}^{-1}`` that turns a weak-form operator into
a bracket on the degrees of freedom, formed through the mass *factorisation* rather than
through [`inverse_mass_matrix`](@ref).

``\mathbb{M}`` is symmetric, so the second factor is the same column solve applied to the
transpose. On a [`TensorSplineSpace`](@ref) that is ``D`` one-dimensional solves per column
against the ``N^2`` storage and ``O(N^3)`` product an explicit inverse would cost twice over.

A space names its own mass matrix, which is the pairing every bracket on an unmapped domain
uses. The second form takes the factorisation directly, for a bracket whose pairing is a
mapped one and therefore not the space's.
"""
function _mass_sandwich(F, A::AbstractMatrix)
    permutedims(_mass_solve(F, permutedims(_mass_solve(F, A))))
end

function _mass_sandwich(s::DiscreteSpace, A::AbstractMatrix)
    _mass_sandwich(mass_factorization(s), A)
end

function _mass_solve(F, A::AbstractMatrix{T}) where {T}
    X = Matrix{T}(undef, size(A))
    for j in axes(A, 2)
        # densified per column: the Kronecker solve reshapes its right-hand side, which a
        # sparse column does not support, and the result is dense in any case
        X[:, j] = F \ Vector(A[:, j])
    end
    return X
end

@doc raw"""
    _mass_apply(space, v)

The product ``\mathbb{M} v``, formed through the space's mass *operator* wherever it has one
rather than through the assembled matrix.

`mass_matrix` is a stored constant on a `SplineSpace` and a `LagrangeSpace`, so the generic
method below costs nothing there. On a [`TensorSplineSpace`](@ref) it is not stored: the matrix
is the Kronecker product of the per-axis factors, it is rebuilt on every call, and
[`ProjectorBracket`](@ref) needs ``\mathbb{M} \hat{\phi}`` on every single contraction. The
`KroneckerMass` forms the same product as ``D`` one-dimensional applications without
assembling anything, which is what keeps [`metric_apply`](@ref) independent of the mesh in
cost as well as in algebra.
"""
_mass_apply(s::DiscreteSpace, v::AbstractVector) = mass_matrix(s) * v

_mass_apply(s::TensorSplineSpace, v::AbstractVector) = mass_operator(s) * v

## Double bracket

@doc raw"""
    DoubleBracket(space, ĥ)
    DoubleBracket(space, Λ)

The metric double bracket of a two-dimensional field,

```math
(F, G) = \int_\Omega
    \left[ \frac{\delta F}{\delta u}, h \right]_J
    \left[ \frac{\delta G}{\delta u}, h \right]_J dx ,
\qquad [f, g]_J = \partial_1 f \, \partial_2 g - \partial_2 f \, \partial_1 g ,
```

with ``h = \delta H / \delta u``. Since ``[f, h]_J = \nabla f \cdot X_h`` with the
Hamiltonian vector field ``X_h = (\partial_2 h, -\partial_1 h)``, the bracket is

```math
(F, G) = \int_\Omega \nabla \frac{\delta F}{\delta u} \cdot
         \left( X_h \otimes X_h \right) \nabla \frac{\delta G}{\delta u} \, dx ,
```

a variable-**tensor**-coefficient stiffness matrix — [`tensor_weighted_matrix`](@ref) with
``\mathbb{D} = X_h \otimes X_h`` — sandwiched between two inverse mass matrices, which is
what the discrete functional derivative produces. The flow it generates is
``\partial_t u = \nabla \cdot (X_h \otimes X_h \, \nabla u)``, diffusion along the contours
of ``h`` and none across them.

The coefficient is written as the outer product rather than as
``Q_2(\nabla h) = |\nabla h|^2 \mathbb{I} - \nabla h \otimes \nabla h``, which in two
dimensions is the same matrix exactly — `scripts/verify_metric_collapse.jl` checks the
identity over five decades of scale. The outer product is the form that is manifestly a Gram
matrix, hence manifestly positive semi-definite, and it is the ``\perp`` in it that carries
the degeneracy: ``X_h \cdot \nabla h = \partial_2 h \, \partial_1 h - \partial_1 h \,
\partial_2 h`` vanishes *pointwise*, so ``\mathbb{G} \, \partial H / \partial \hat{u} = 0``
holds at the quadrature level and not merely in the continuum.

# The two forms of `h`

`ĥ` is a coefficient vector when ``h`` is prescribed, and `Λ` is the matrix of the linear map
``\hat{h} = \Lambda \hat{u}`` when it is not. The manuscript's two-dimensional examples need
exactly these: ``h = \cos^2 x_1 \sin^2 x_2`` fixed for parallel diffusion, and
``\hat{\psi} = \mathbb{K}^{-1} \mathbb{M} \hat{\omega}`` from the elliptic solve for reduced
Euler. Both make [`metric_derivative`](@ref) analytic; a general nonlinear ``h`` would not,
and is not accepted rather than being differenced silently.

```jldoctest
julia> s = TensorSplineSpace((6, 6), 3);

julia> b = DoubleBracket(s, project(s, x -> cos(x[1])^2 * sin(x[2])^2));

julia> issymmetric(b, zeros(36)), degeneracy_residual(b, zeros(36)) < 1e-13
(true, true)
```
"""
struct DoubleBracket{T, ST <: TensorSplineSpace{T, 2}, HT <: AbstractVecOrMat{T}} <:
       MetricBracket{T}
    space::ST
    h::HT

    function DoubleBracket(s::ST, h::HT) where {
            T, ST <: TensorSplineSpace{T, 2}, HT <: AbstractVecOrMat{T}}
        size(h, 1) == nbasis(s) || throw(DimensionMismatch(
            "the generating field has $(size(h, 1)) rows but the space has $(nbasis(s)) " *
            "basis functions"))
        h isa AbstractMatrix && size(h, 2) != nbasis(s) &&
            throw(DimensionMismatch(
                "the generating map is $(size(h)) but the space has $(nbasis(s)) basis functions"))
        new{T, ST, HT}(s, h)
    end
end

Base.size(b::DoubleBracket) = (nbasis(b.space), nbasis(b.space))
space(b::DoubleBracket) = b.space

@doc raw"""
    hamiltonian_field(space, ĥ)

The Hamiltonian vector field ``X_h = (\partial_2 h, -\partial_1 h)`` sampled at the
quadrature points, as a pair of vectors.

This is the ``\perp`` of ``\nabla h``, and the only place in [`DoubleBracket`](@ref) where the
sign convention is fixed. It cancels out of the coefficient — ``X_h \otimes X_h`` is
unchanged by ``X_h \mapsto -X_h`` — but not out of the degeneracy, which is what
[`degeneracy_residual`](@ref) exists to notice.
"""
function hamiltonian_field(s::TensorSplineSpace{T, 2}, ĥ::AbstractVector) where {T}
    (field(s, ĥ, (0, 1)), -field(s, ĥ, (1, 0)))
end

# (a ⊗ b)_kl = a_k b_l at the quadrature points, in the component form
# `tensor_weighted_matrix` takes. Built element by element rather than as `[a b; c d]`,
# which concatenates the sample vectors into one long matrix of numbers instead.
function _outer(a::NTuple{2, V}, b::NTuple{2, V}) where {V <: AbstractVector}
    𝔻 = Matrix{V}(undef, 2, 2)
    for l in 1:2, k in 1:2

        𝔻[k, l] = a[k] .* b[l]
    end
    return 𝔻
end

"""
    metric_operator(b::DoubleBracket, û)

The weak-form operator ``\\mathbb{A}_{KL} = \\int_\\Omega \\partial_k \\Phi_K
(X_h \\otimes X_h)_{kl} \\partial_l \\Phi_L \\, dx``, without the surrounding inverse mass
matrices.

Sparse, unlike [`metric_matrix`](@ref): the densification is entirely in the sandwich.
"""
function metric_operator(b::DoubleBracket, û::AbstractVector)
    X = hamiltonian_field(b.space, _generator(b, û))
    tensor_weighted_matrix(b.space, _outer(X, X))
end

function metric_matrix(b::DoubleBracket, û::AbstractVector)
    _mass_sandwich(b.space, metric_operator(b, û))
end

@doc raw"""
    metric_apply(b::DoubleBracket, û, c)

``\mathbb{G} c`` formed without assembling either the operator or the sandwich.

With ``v = \mathbb{M}^{-1} c`` the contraction is
``\int_\Omega \partial_k \Phi_K \, X_k \, (X \cdot \nabla v_h) \, dx``, which costs
``O(NQ)`` against the ``O(N^2 Q)`` of assembling the operator first and the ``O(N^3)`` of the
sandwich. This is the path a time loop takes.
"""
function metric_apply(b::DoubleBracket, û::AbstractVector, c::AbstractVector)
    s = b.space
    w = quadrature_weights(s)
    F = mass_factorization(s)
    X = hamiltonian_field(s, _generator(b, û))

    v = F \ Vector(c)
    # X · ∇v_h at the quadrature points, then tested against ∂_k Φ_K weighted by X_k
    Xv = sum(X[l] .* field(s, v, _unit_index(2, l)) for l in 1:2)
    Av = sum(basis_values(s, _unit_index(2, k)) * (w .* X[k] .* Xv) for k in 1:2)
    F \ Av
end

function metric_derivative(b::DoubleBracket{T, ST, <:AbstractVector}, û::AbstractVector) where {
        T, ST}
    N = nbasis(b.space)
    zeros(T, N, N, N)
end

@doc raw"""
    metric_derivative(b::DoubleBracket, û)

For a `Λ`-generated bracket, ``\mathbb{D} = X_h \otimes X_h`` is quadratic in ``\hat{h}`` and
``\hat{h}`` is linear in ``\hat{u}``, so

```math
\frac{\partial \mathbb{D}}{\partial \hat{u}_m}
    = \dot{X}^m \otimes X_h + X_h \otimes \dot{X}^m ,
```

with ``\dot{X}^m`` the Hamiltonian vector field of the ``m``-th column of ``\Lambda``. Each
column is one assembly and one sandwich, so this is ``N`` of both — off the time loop by
construction, and the reason a metriplectic flow built on it wants the directional derivative
rather than the tensor.
"""
function metric_derivative(b::DoubleBracket{T, ST, <:AbstractMatrix}, û::AbstractVector) where {
        T, ST}
    s = b.space
    N = nbasis(s)
    X = hamiltonian_field(s, _generator(b, û))

    dG = zeros(T, N, N, N)
    for m in 1:N
        Ẋ = hamiltonian_field(s, Vector(b.h[:, m]))
        Ȧ = tensor_weighted_matrix(s, _outer(Ẋ, X) .+ _outer(X, Ẋ))
        dG[m, :, :] = _mass_sandwich(s, Ȧ)
    end
    return dG
end

function metric_directional(b::DoubleBracket{T, ST, <:AbstractVector}, û::AbstractVector,
        v::AbstractVector) where {T, ST}
    zeros(T, length(û), length(û))
end

@doc raw"""
    metric_directional(b::DoubleBracket, û, v)

The same ``N`` perturbed assemblies as [`metric_derivative`](@ref), but each one contracted
against ``\mathbb{M}^{-1} v`` and solved once rather than sandwiched.

What that removes is the ``N`` sandwiches — ``O(N^3)`` each, so ``O(N^4)`` in total — and what
it cannot remove is the ``N`` perturbed assemblies, which both paths pay. So the saving here is
a **constant factor and not an order**: the assemblies dominate at every size a
two-dimensional problem reaches, and claiming an order would be an extrapolation rather than a
measurement. Contrast [`metric_directional`](@ref)`(::ProjectorBracket, …)`, which assembles
nothing and does save an order. `scripts/verify_metriplectic_flow.jl` measures both ratios;
they are wall-clock numbers and are reported there rather than pinned here, where they would
go stale on the next machine.

Never *slower*, though, and that is the claim that is load-bearing: this runs once per Newton
iteration and the tensor cannot.
"""
function metric_directional(b::DoubleBracket{T, ST, <:AbstractMatrix}, û::AbstractVector,
        v::AbstractVector) where {T, ST}
    s = b.space
    N = nbasis(s)
    F = mass_factorization(s)
    X = hamiltonian_field(s, _generator(b, û))
    w = F \ Vector(v)                      # the inner M⁻¹ of the sandwich, formed once

    D = zeros(T, N, N)
    for m in 1:N
        Ẋ = hamiltonian_field(s, Vector(b.h[:, m]))
        Ȧ = tensor_weighted_matrix(s, _outer(Ẋ, X) .+ _outer(X, Ẋ))
        D[:, m] = F \ Vector(Ȧ * w)
    end
    return D
end

## Projector bracket

@doc raw"""
    ProjectorBracket(space, φ̂)
    ProjectorBracket(space, Λ)

The metric bracket built from the ``L^2`` projector orthogonal to ``\phi = \delta H /
\delta u``,

```math
(F, G) = \left( \frac{\delta F}{\delta u}, \, \Pi_H \frac{\delta G}{\delta u} \right)_{L^2} ,
\qquad
\Pi_H v = v - \frac{(\phi, v)_{L^2}}{\| \phi \|^2_{L^2}} \, \phi .
```

On the degrees of freedom this is a **rank-one correction to the inverse mass matrix**,

```math
\mathbb{G} = \mathbb{M}^{-1}
    - \frac{\hat{\phi} \, \hat{\phi}^T}{\hat{\phi}^T \mathbb{M} \hat{\phi}} ,
```

and needs no quadrature at all: the two inverse mass matrices of the discrete functional
derivative cancel against the ``\mathbb{M}`` of the ``L^2`` product, and what is left is
algebra in ``\hat{\phi}``. [`metric_apply`](@ref) is therefore one mass solve, one mass apply
and two inner products — no quadrature loop, and nothing of size ``N^2`` formed.

Both mass operations go through the space's own operator, `mass_factorization` for the solve
and [`_mass_apply`](@ref) for the product, so neither assembles ``\mathbb{M}``. That is what
makes the cost independent of the mesh in fact and not only in algebra: on a
[`TensorSplineSpace`](@ref) the call allocates 2 400 B at ``N = 25``, 4 576 B at ``N = 64`` and
9 056 B at ``N = 144`` — the vectors it returns, growing like ``N``. Reaching for
`mass_matrix` instead would rebuild the Kronecker product every time, which is 10 640, 29 376
and 66 944 B of that same call.

Positive semi-definiteness is Cauchy-Schwarz in the ``\mathbb{M}``-inner product: with
``v = \mathbb{M}^{-1} c``,

```math
c^T \mathbb{G} c = (v, v)_\mathbb{M}
    - \frac{(\hat{\phi}, v)^2_\mathbb{M}}{(\hat{\phi}, \hat{\phi})_\mathbb{M}} \ge 0 ,
```

with equality exactly on the span of ``\hat{\phi}``. The degeneracy is the same statement
read the other way: ``\mathbb{G} \, \mathbb{M} \hat{\phi} = \hat{\phi} - \hat{\phi} = 0``
identically, in exact arithmetic, at any resolution.

`φ̂` and `Λ` are the two forms of the generating field described under
[`DoubleBracket`](@ref), and the space may be of any dimension — nothing here is
two-dimensional.

```jldoctest
julia> s = SplineSpace(16, 3);

julia> b = ProjectorBracket(s, project(s, sin));

julia> ispositive_semidefinite(b, zeros(16)), degeneracy_residual(b, zeros(16)) < 1e-13
(true, true)
```
"""
struct ProjectorBracket{T, ST <: DiscreteSpace{T}, HT <: AbstractVecOrMat{T}} <:
       MetricBracket{T}
    space::ST
    h::HT

    function ProjectorBracket(s::ST, h::HT) where {
            T, ST <: DiscreteSpace{T}, HT <: AbstractVecOrMat{T}}
        size(h, 1) == nbasis(s) || throw(DimensionMismatch(
            "the generating field has $(size(h, 1)) rows but the space has $(nbasis(s)) " *
            "basis functions"))
        h isa AbstractMatrix && size(h, 2) != nbasis(s) &&
            throw(DimensionMismatch(
                "the generating map is $(size(h)) but the space has $(nbasis(s)) basis functions"))
        new{T, ST, HT}(s, h)
    end
end

Base.size(b::ProjectorBracket) = (nbasis(b.space), nbasis(b.space))
space(b::ProjectorBracket) = b.space

@doc raw"""
    project_orthogonal(b::ProjectorBracket, û, v̂)

The projector ``\Pi_H`` applied to the field with coefficients `v̂`, in coefficients.

``\Pi_H \hat{\phi} = 0`` and ``\Pi_H^2 = \Pi_H``; both are what make the bracket degenerate
on the Hamiltonian and idempotent, and both are properties of this map rather than of the
assembled matrix.
"""
function project_orthogonal(b::ProjectorBracket, û::AbstractVector, v̂::AbstractVector)
    φ = _generator(b, û)
    Mφ = _mass_apply(b.space, φ)
    v̂ - φ * (dot(Mφ, v̂) / dot(Mφ, φ))
end

function metric_matrix(b::ProjectorBracket, û::AbstractVector)
    φ = _generator(b, û)
    n = dot(_mass_apply(b.space, φ), φ)
    inverse_mass_matrix(b.space) - (φ * φ') / n
end

function metric_apply(b::ProjectorBracket, û::AbstractVector, c::AbstractVector)
    φ = _generator(b, û)
    n = dot(_mass_apply(b.space, φ), φ)
    (mass_factorization(b.space) \ Vector(c)) - φ * (dot(φ, c) / n)
end

function metric_derivative(b::ProjectorBracket{T, ST, <:AbstractVector}, û::AbstractVector) where {
        T, ST}
    N = nbasis(b.space)
    zeros(T, N, N, N)
end

@doc raw"""
    metric_derivative(b::ProjectorBracket, û)

For a `Λ`-generated bracket, with ``\hat{\phi} = \Lambda \hat{u}`` and
``n = \hat{\phi}^T \mathbb{M} \hat{\phi}``,

```math
\frac{\partial \mathbb{G}_{ij}}{\partial \hat{u}_m}
    = - \frac{\Lambda_{im} \hat{\phi}_j + \hat{\phi}_i \Lambda_{jm}}{n}
      + \frac{2 \, \hat{\phi}_i \hat{\phi}_j \, (\Lambda^T \mathbb{M} \hat{\phi})_m}{n^2} ,
```

the ``\mathbb{M}^{-1}`` term being constant. No quadrature and no assembly: this is
``O(N^3)`` arithmetic on two vectors and a matrix.
"""
function metric_derivative(b::ProjectorBracket{T, ST, <:AbstractMatrix}, û::AbstractVector) where {
        T, ST}
    s = b.space
    N = nbasis(s)
    Λ = b.h
    φ = _generator(b, û)
    Mφ = mass_matrix(s) * φ
    n = dot(Mφ, φ)
    q = Λ' * Mφ

    dG = zeros(T, N, N, N)
    @inbounds for j in 1:N, i in 1:N, m in 1:N
        dG[m, i, j] = -(Λ[i, m] * φ[j] + φ[i] * Λ[j, m]) / n +
                      2 * φ[i] * φ[j] * q[m] / n^2
    end
    return dG
end

function metric_directional(
        b::ProjectorBracket{T, ST, <:AbstractVector}, û::AbstractVector,
        v::AbstractVector) where {T, ST}
    zeros(T, length(û), length(û))
end

@doc raw"""
    metric_directional(b::ProjectorBracket, û, v)

Closed form, ``O(N^2)``: contracting the tensor of [`metric_derivative`](@ref) with `v`
collapses every index sum into an inner product,

```math
D_{im} = - \frac{(\hat{\phi} \cdot v) \, \Lambda_{im}
              + \hat{\phi}_i \, (\Lambda^T v)_m}{n}
         + \frac{2 \, (\hat{\phi} \cdot v) \, \hat{\phi}_i \,
                 (\Lambda^T \mathbb{M} \hat{\phi})_m}{n^2} ,
\qquad n = \hat{\phi}^T \mathbb{M} \hat{\phi} .
```

Nothing of size ``N^3`` is formed, no quadrature is touched and nothing is assembled — this is
two matrix-vector products and two dot products. It is the one bracket for which contracting
first saves an **order of magnitude** rather than a constant factor, and the saving grows with
``N`` because there is no assembly left to dominate the sandwich it removes.
`scripts/verify_metriplectic_flow.jl` measures the ratio; it is a wall-clock number and is
reported there rather than pinned here, where it would go stale on the next machine.
"""
function metric_directional(
        b::ProjectorBracket{T, ST, <:AbstractMatrix}, û::AbstractVector,
        v::AbstractVector) where {T, ST}
    Λ = b.h
    φ = _generator(b, û)
    Mφ = _mass_apply(b.space, φ)
    n = dot(Mφ, φ)
    φv = dot(φ, v)
    (-φv / n) .* Matrix(Λ) .- (φ / n) * (Λ' * v)' .+
    ((2 * φv / n^2) .* φ) * (Λ' * Mφ)'
end
