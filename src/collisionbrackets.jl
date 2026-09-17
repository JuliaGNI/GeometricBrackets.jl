
@doc raw"""
    CollisionBracket(space, φ̂; mobility = 1, mobility_derivative = nothing, density = 1)
    CollisionBracket(space, Λ; mobility = 1, mobility_derivative = nothing, density = 1)

The collision-like metric bracket of a two-dimensional field,

```math
(F, G) = \frac{1}{2} \int_\Omega \! \int_\Omega
    \kappa(x, x') \,
    \big( \nabla f(x) - \nabla f(x') \big)^T
    Q_2 \big( \nabla \phi(x) - \nabla \phi(x') \big)
    \big( \nabla g(x) - \nabla g(x') \big) \, d\mu(x') \, d\mu(x) ,
```

with ``f = \delta F / \delta u``, ``g = \delta G / \delta u``, ``\phi = \delta H / \delta u``,
the factorised kernel ``\kappa(x, x') = M(x, u(x)) \, M(x', u(x'))`` and
``Q_2(z) = |z|^2 \mathbb{I} - z \otimes z``.

Degeneracy is structural and needs no cancellation between distant terms: setting ``G = H``
makes the last factor ``\nabla \phi(x) - \nabla \phi(x') = z``, and ``Q_2(z) \, z = 0``
identically, at every *pair* of quadrature points. Positive semi-definiteness is equally
structural — ``Q_2`` is positive semi-definite and ``\kappa > 0``, so ``(F, F) \ge 0``.

# The collapse, which is what makes it affordable

Written as it stands, every outer quadrature node carries a full inner quadrature and one
assembly costs ``O(N_q^2)`` — some ``10^{10}`` pair evaluations at ``64^2`` cells. It does not
have to. In two dimensions ``Q_2(z) = z^\perp \otimes z^\perp`` **exactly**, with
``z^\perp = (-z_2, z_1)``, and perping is linear, so with ``\beta = (\nabla \phi)^\perp`` the
kernel is ``(\beta(x) - \beta(x')) \otimes (\beta(x) - \beta(x'))`` — a *quadratic* in the
inner point. The inner integral is therefore a fixed set of global moments, evaluated once,
after which everything is pointwise-local in ``x``. Because one fixed global quadrature rule
serves the inner sum at every outer point, the collapse is exact **at the quadrature level**
and perturbs no discrete conservation property.

The moments are accumulated in the **recentred** variables ``\gamma = \beta - \bar\beta``,
``\bar\beta = m_0^{-1} \int \beta M \, d\mu``:

```math
m_0 = \int M d\mu , \qquad
q_1 = \int \gamma \, M d\mu \; (= 0) , \qquad
\Sigma = \int \gamma \otimes \gamma \, M d\mu ,
```

and the diffusion tensor of the manuscript is

```math
\mathbb{D}_s(u; x) = \int_\Omega Q_2 \big( \nabla \phi(x) - \nabla \phi(x') \big) M \, d\mu'
    = m_0 \, \gamma \otimes \gamma - \gamma \otimes q_1 - q_1 \otimes \gamma + \Sigma .
```

Recentring is a **correctness requirement, not a refinement**. ``\Sigma`` is accumulated as
``\int (\beta - \bar\beta) \otimes (\beta - \bar\beta) M d\mu'`` directly; forming it instead
as ``M_2 - m_0 \bar\beta \otimes \bar\beta`` reinstates exactly the cancellation the centring
removes, and with it the amplification ``|\alpha|^2 m_0 / \| \mathbb{D}_s \|`` that blows up
when ``\nabla \phi`` has a large mean and a small variation — a plausible relaxed
Grad-Shafranov state. Measured in `scripts/verify_metric_collapse.jl`: at a gradient spread of
``10^{-7}`` the uncentred form's relative error passes 1 and it goes indefinite in 180 of 200
draws, while the centred form stays positive semi-definite in 1800 of 1800.

The ``q_1`` terms are kept even though ``q_1`` vanishes at the centre. They cost nothing, they
make ``\mathbb{D}_s`` correct for *any* origin — the bracket depends only on differences
``\beta(x) - \beta(x')``, so the origin is free — and it is that freedom which makes
[`metric_derivative`](@ref) analytic: the centre may be frozen while ``\hat{u}`` moves, and
``\partial \bar\beta / \partial \hat{u}`` never appears.

# Where the collapse would break

Three hypotheses, none of them cosmetic. ``Q`` must be **polynomial** in ``z`` — it fails for
the true Landau kernel ``|z|^{-3} Q_2(z)``; ``\kappa`` must **factorise**; and ``d\mu(x')``
must **not depend on** ``x``. The Grad-Shafranov weight ``d\mu = dr \, dz / r`` is harmless
precisely because it is of the third kind: it is absorbed into the quadrature weights and
never sees the outer point. That is why `density` is an explicit field of this type rather
than an assumption — a measure is either admissible or it is not, and the type says which one
was used.

# The entropy enters only through `M`

``M`` is fixed by the entropy density through `eq:M-condition`, ``M \, \partial_y^2 s = 1``,
so the manuscript's three cases are three mobilities:

| ``s(x, y)`` | ``M`` | `mobility` |
|:--|:--|:--|
| ``y^2/2`` (§5.4, reduced Euler) | ``1`` | `1` — the default |
| ``y \log y`` (§5.4, Gibbs) | ``y`` | `(x, u) -> u`, `mobility_derivative = (x, u) -> 1` |
| ``y^2 / 2(Cr^2+D)`` (§5.5, Grad-Shafranov) | ``Cr^2 + D`` | `(x, u) -> C*x[1]^2 + D`, `mobility_derivative = 0` |

`eq:M-condition` also requires ``M > 0``, which is what makes ``\kappa`` a positive kernel;
nothing here enforces it, and a mobility that changes sign gives a bracket that is not
positive semi-definite, which [`ispositive_semidefinite`](@ref) will say.

The other half of the entropy, ``\partial_x \partial_y s``, does **not** appear: it enters
``\delta S / \delta u``, which is the flow's business, not the bracket's. What the bracket
contracts against is an arbitrary vector, and the manuscript's

```math
\mathbb{F}_s(u; x) = \int_\Omega Q_2 \big( \nabla \phi(x) - \nabla \phi(x') \big) \,
    w(x') \, d\mu' , \qquad w = M \nabla \frac{\delta S}{\delta u}
    = \nabla u + M \, \partial_x \partial_y s ,
```

is the ``G = S`` case of [`metric_apply`](@ref), where the second equality is `eq:M-condition`
again.

A `mobility` given as a function must be given its derivative ``\partial M / \partial u`` as
`mobility_derivative` — `0` where, as in §5.5, ``M`` depends on ``x`` alone. It is not
differenced silently, for the same reason a general nonlinear generating field is refused by
[`DoubleBracket`](@ref): what makes the Jacobian of a metriplectic flow analytic is exactly
that this derivative is known. `φ̂` and `Λ` are the two forms of the generating field
documented there.

A §5.5 note that is easy to lose: because ``M`` is there independent of
``u``, ``\mathbb{G}`` is *quadratic* in ``\hat{u}`` and ``\partial S / \partial \hat{u}`` is
*linear*, so the dissipative residual is an exact **cubic polynomial** in the degrees of
freedom. Its Jacobian is analytic and cheap, and finite differences have no business being
used for it.

```jldoctest
julia> s = TensorSplineSpace((4, 4), 2);

julia> M = Matrix(mass_matrix(s));

julia> b = CollisionBracket(s, (Matrix(stiffness_matrix(s)) + 0.5M) \ M);

julia> û = project(s, p -> sin(p[1]) * cos(p[2]));

julia> issymmetric(b, û), degeneracy_residual(b, û) < 1e-12
(true, true)
```
"""
struct CollisionBracket{
    T, ST <: PlanarSplineSpace{T}, HT <: AbstractVecOrMat{T}, MF, MD} <:
       MetricBracket{T}
    space::ST
    h::HT
    mobility::MF
    mobility_derivative::MD
    density::Vector{T}
end

function CollisionBracket(s::PlanarSplineSpace{T}, h::AbstractVecOrMat{T};
        mobility = one(T), mobility_derivative = nothing, density = one(T)) where {T}
    size(h, 1) == nbasis(s) || throw(DimensionMismatch(
        "the generating field has $(size(h, 1)) rows but the space has $(nbasis(s)) " *
        "basis functions"))
    h isa AbstractMatrix && size(h, 2) != nbasis(s) &&
        throw(DimensionMismatch(
            "the generating map is $(size(h)) but the space has $(nbasis(s)) basis functions"))
    M, dM = _mobility_pair(T, mobility, mobility_derivative)
    ρ = _density_samples(s, density)
    CollisionBracket{T, typeof(s), typeof(h), typeof(M), typeof(dM)}(s, h, M, dM, ρ)
end

Base.size(b::CollisionBracket) = (nbasis(b.space), nbasis(b.space))
space(b::CollisionBracket) = b.space

# A number stands for the constant function of that value, which is the whole of §5.4's
# `M = 1` and of `∂M/∂u = 0` wherever `M` depends on `x` alone, as it does in §5.5.
_as_mobility(::Type{T}, m::Number) where {T} = (v = T(m); (x, u) -> v)
_as_mobility(::Type{T}, m) where {T} = m

function _mobility_pair(::Type{T}, mobility, mobility_derivative) where {T}
    mobility isa Number || mobility_derivative !== nothing ||
        throw(ArgumentError(
            "a mobility given as a function needs its derivative: pass `mobility_derivative` " *
            "as (x, u) -> ∂M/∂u, or as `0` when M depends on x alone. It is not differenced " *
            "silently, because the Jacobian of a metriplectic flow is analytic exactly when " *
            "this one is"))
    (_as_mobility(T, mobility),
        _as_mobility(T, something(mobility_derivative, zero(T))))
end

function _density_samples(s::PlanarSplineSpace{T}, ρ::Number) where {T}
    fill(T(ρ), length(quadrature_weights(s)))
end

function _density_samples(s::PlanarSplineSpace{T}, ρ::AbstractVector) where {T}
    length(ρ) == length(quadrature_weights(s)) || throw(DimensionMismatch(
        "the measure density was sampled at $(length(ρ)) points but the quadrature grid " *
        "has $(length(quadrature_weights(s)))"))
    Vector{T}(ρ)
end

function _density_samples(s::PlanarSplineSpace{T}, ρ) where {T}
    T[ρ(x) for x in quadrature_nodes(s)]
end

@doc raw"""
    _collision_state(bracket, û)

Everything the assembly reads off the state, in one pass over the quadrature grid: the
mobility `M` and its derivative `Mu`, the measure weights ``\mu = \rho \, w``, the kernel
weights ``c = M \mu``, the recentred ``\gamma = (\nabla \phi)^\perp - \bar\beta``, and the six
scalar moments ``m_0``, ``q_1`` and ``\Sigma``.

The ``\perp`` is fixed here and nowhere else: ``\beta = (-\partial_2 \phi, \partial_1 \phi)``.
It is what carries the degeneracy, and dropping it leaves a bracket that is still symmetric
and still positive semi-definite — see [`degeneracy_residual`](@ref).
"""
function _collision_state(b::CollisionBracket{T}, û::AbstractVector) where {T}
    s = b.space
    x = quadrature_nodes(s)
    u = field(s, û, (0, 0))
    M = T[b.mobility(x[r], u[r]) for r in eachindex(u)]
    Mu = T[b.mobility_derivative(x[r], u[r]) for r in eachindex(u)]
    μ = quadrature_weights(s) .* b.density
    c = M .* μ

    φ̂ = _generator(b, û)
    β = (-field(s, φ̂, (0, 1)), field(s, φ̂, (1, 0)))

    m₀ = sum(c)
    β̄ = ntuple(k -> dot(c, β[k]) / m₀, 2)
    γ = ntuple(k -> β[k] .- β̄[k], 2)

    (; M, Mu, μ, c, γ, _kernel_moments(c, γ)...)
end

# The six moments of the D_s side, accumulated centred: Σ is ∫γ⊗γ M dμ' *directly*, never
# M₂ - m₀ β̄⊗β̄, which is the cancellation the centring exists to remove.
function _kernel_moments(c::AbstractVector{T}, γ::NTuple{2, <:AbstractVector}) where {T}
    m₀ = sum(c)
    q₁ = ntuple(k -> dot(c, γ[k]), 2)
    Σ = T[dot(c, γ[k] .* γ[l]) for k in 1:2, l in 1:2]
    (; m₀, q₁, Σ)
end

@doc raw"""
    _diffusion_tensor(bracket, û)

The manuscript's ``\mathbb{D}_s(u; x)``, sampled on the quadrature grid as a ``2 \times 2``
matrix of vectors — the component form [`tensor_weighted_matrix`](@ref) takes.

Index convention, load-bearing: ``(a \otimes b)_{kl} = a_k b_l`` and
``(\mathbb{A} v)_k = \mathbb{A}_{kl} v_l``.
"""
_diffusion_tensor(b::CollisionBracket, û::AbstractVector) = _diffusion_tensor(_collision_state(b, û))

function _diffusion_tensor(st)
    γ, m₀, q₁, Σ = st.γ, st.m₀, st.q₁, st.Σ
    𝔻 = Matrix{typeof(γ[1])}(undef, 2, 2)
    for l in 1:2, k in 1:2

        𝔻[k, l] = @. m₀ * γ[k] * γ[l] - γ[k] * q₁[l] - q₁[k] * γ[l] + Σ[k, l]
    end
    return 𝔻
end

# The derivative of `_diffusion_tensor` in the direction (δγ, δc), by the product rule. The
# centre is frozen: D_s depends on β only through differences β(x) - β(x'), so the origin is
# free and δβ̄ is not part of the derivative.
function _diffusion_tensor_derivative(st, δγ, δc)
    γ, m₀, q₁ = st.γ, st.m₀, st.q₁
    δm₀, δq₁, δΣ = _kernel_moments_derivative(st.c, γ, δc, δγ)
    δ𝔻 = Matrix{typeof(γ[1])}(undef, 2, 2)
    for l in 1:2, k in 1:2

        δ𝔻[k, l] = @. δm₀ * γ[k] * γ[l] + m₀ * (δγ[k] * γ[l] + γ[k] * δγ[l]) -
                      δγ[k] * q₁[l] - γ[k] * δq₁[l] - δq₁[k] * γ[l] - q₁[k] * δγ[l] +
                      δΣ[k, l]
    end
    return δ𝔻
end

function _kernel_moments_derivative(c, γ, δc, δγ)
    δm₀ = sum(δc)
    δq₁ = ntuple(k -> dot(δc, γ[k]) + dot(c, δγ[k]), 2)
    δΣ = [dot(δc, γ[k] .* γ[l]) + dot(c, δγ[k] .* γ[l] .+ γ[k] .* δγ[l])
          for k in 1:2, l in 1:2]
    (δm₀, δq₁, δΣ)
end

@doc raw"""
    _cross_factors(space, c, γ)

The eight ``N``-vectors through which the **nonlocal** half of the operator factorises,

```math
\mathbb{S}_{Kj} = \int c \, \partial_j \Phi_K , \quad
\mathbb{T}_{Kij} = \int c \, \partial_i \Phi_K \gamma_j , \quad
\mathbb{R}_{Kj} = \int c \, (\nabla \Phi_K \cdot \gamma) \, \gamma_j ,
```

with ``a_K = \mathbb{T}_{K11} + \mathbb{T}_{K22}`` derived rather than accumulated.

This is the operator-level counterpart of the fourteen scalar moments. Fourteen scalars are
enough for the *pointwise* coefficients ``\mathbb{D}_s`` and ``\mathbb{F}_s``, which is what
[`metric_apply`](@ref) evaluates; an assembled matrix is not a pointwise coefficient, and the
cross term ``\int \! \int \kappa \, \nabla \Phi_K(x)^T Q_2 \nabla \Phi_L(x') `` needs one
``N``-vector per separable factor instead. There are nine such factors and eight
accumulators — rank at most nine, independent of the mesh.
"""
function _cross_factors(s::PlanarSplineSpace, c::AbstractVector, γ)
    P = (basis_values(s, (1, 0)), basis_values(s, (0, 1)))
    S = ntuple(j -> P[j] * c, 2)
    𝕋 = [P[i] * (c .* γ[j]) for i in 1:2, j in 1:2]
    R = ntuple(j -> sum(P[i] * (c .* γ[i] .* γ[j]) for i in 1:2), 2)
    (; S, 𝕋, a = 𝕋[1, 1] .+ 𝕋[2, 2], R)
end

# The cross term as a bilinear form in the factors, so that its derivative is the
# polarisation `_cross_operator(δf, f) + _cross_operator(f, δf)` and nothing has to be
# expanded twice.  (b ⊗ w)_ij = b_i w_j throughout.
function _cross_operator(f, g)
    C = f.R[1] * g.S[1]' .+ f.R[2] * g.S[2]' .+ f.S[1] * g.R[1]' .+ f.S[2] * g.R[2]' .-
        f.a * g.a'
    for j in 1:2, i in 1:2

        C .-= f.𝕋[i, j] * g.𝕋[j, i]'
    end
    return C
end

"""
    metric_operator(b::CollisionBracket, û)

The weak-form operator ``\\mathbb{A}_{KL}``, without the surrounding inverse mass matrices:
the local, tensor-coefficient stiffness matrix built on ``M \\mathbb{D}_s`` minus the
nonlocal cross term of [`_cross_factors`](@ref).

Dense, unlike the [`DoubleBracket`](@ref) operator: the cross term is a rank-nine
correction and a bracket that is nonlocal in space cannot be sparse.
"""
function metric_operator(b::CollisionBracket, û::AbstractVector)
    _collision_operator(b, _collision_state(b, û))
end

function _collision_operator(b::CollisionBracket{T}, st) where {T}
    s = b.space
    𝔻 = _diffusion_tensor(st)
    ϱ = b.density .* st.M
    coefficient = Matrix{Vector{T}}(undef, 2, 2)
    for l in 1:2, k in 1:2

        coefficient[k, l] = ϱ .* 𝔻[k, l]
    end
    f = _cross_factors(s, st.c, st.γ)
    Matrix(tensor_weighted_matrix(s, coefficient)) .- _cross_operator(f, f)
end

function _collision_operator_derivative(b::CollisionBracket{T}, st, δγ, δc, δM) where {T}
    s = b.space
    𝔻 = _diffusion_tensor(st)
    δ𝔻 = _diffusion_tensor_derivative(st, δγ, δc)
    ϱ = b.density .* st.M
    δϱ = b.density .* δM
    coefficient = Matrix{Vector{T}}(undef, 2, 2)
    for l in 1:2, k in 1:2

        coefficient[k, l] = δϱ .* 𝔻[k, l] .+ ϱ .* δ𝔻[k, l]
    end
    f = _cross_factors(s, st.c, st.γ)
    δf = _cross_factors_derivative(s, st.c, st.γ, δc, δγ)
    Matrix(tensor_weighted_matrix(s, coefficient)) .- _cross_operator(δf, f) .-
    _cross_operator(f, δf)
end

function _cross_factors_derivative(s::PlanarSplineSpace, c, γ, δc, δγ)
    P = (basis_values(s, (1, 0)), basis_values(s, (0, 1)))
    S = ntuple(j -> P[j] * δc, 2)
    𝕋 = [P[i] * (δc .* γ[j] .+ c .* δγ[j]) for i in 1:2, j in 1:2]
    R = ntuple(
        j -> sum(P[i] * (δc .* γ[i] .* γ[j] .+ c .* δγ[i] .* γ[j] .+ c .* γ[i] .* δγ[j])
        for i in 1:2), 2)
    (; S, 𝕋, a = 𝕋[1, 1] .+ 𝕋[2, 2], R)
end

function metric_matrix(b::CollisionBracket, û::AbstractVector)
    _mass_sandwich(b.space, metric_operator(b, û))
end

@doc raw"""
    metric_apply(b::CollisionBracket, û, c)

``\mathbb{G} c`` in ``O(N_q)``, through the fourteen scalar moments and without assembling
anything of size ``N^2``.

With ``v = \mathbb{M}^{-1} c`` the field the bracket is contracted against, and
``w = M \nabla v_h`` the manuscript's ``w``,

```math
(\mathbb{A} v)_K = \int_\Omega \nabla \Phi_K \cdot
    \big( \mathbb{D}_s \, w - M \, \mathbb{F}_s \big) \, d\mu ,
```

with ``\mathbb{D}_s`` the six-moment tensor above and ``\mathbb{F}_s`` the eight-moment vector

```math
\mathbb{F}_s = \delta \, (\delta \cdot n_0) - \delta \operatorname{tr} \tilde{B}_c
    - \tilde{B}_c \, \delta + \tilde{T}_c , \qquad \delta = \gamma(x) ,
```

```math
n_0 = \int w \, d\mu' , \qquad
\tilde{B}_c = \int \gamma \otimes w \, d\mu' , \qquad
\tilde{T}_c = \int \gamma \, (\gamma \cdot w) \, d\mu' .
```

Six plus eight is the whole moment count, and it is minimal:
`scripts/verify_metric_collapse.jl` settles sufficiency by perturbing the node data along the
null space of the moment map and necessity by the rank of the moments-to-outputs Jacobian.
``\operatorname{tr} \tilde{B}_c`` is not a fifteenth accumulator, and neither is
``\int |\beta|^2 M d\mu'``.

This is a genuinely independent evaluation of the same operator — the ``\mathbb{F}_s`` route
rather than the ``\mathbb{R}, \mathbb{S}, \mathbb{T}`` factorisation of
[`metric_operator`](@ref) — so agreement with `metric_matrix(b, û) * c` is a cross-check and
not a tautology.
"""
function metric_apply(b::CollisionBracket{T}, û::AbstractVector, c::AbstractVector) where {T}
    s = b.space
    st = _collision_state(b, û)
    𝔻 = _diffusion_tensor(st)
    γ = st.γ

    v̂ = mass_factorization(s) \ Vector(c)
    w = ntuple(k -> st.M .* field(s, v̂, _unit_index(2, k)), 2)   # w = M ∇v_h
    vw = ntuple(k -> st.μ .* w[k], 2)                            # w dμ at the nodes

    n₀ = ntuple(k -> sum(vw[k]), 2)
    B̃ = T[dot(γ[k], vw[l]) for k in 1:2, l in 1:2]               # (γ ⊗ w)_kl = γ_k w_l
    γw = γ[1] .* vw[1] .+ γ[2] .* vw[2]                          # (γ · w) dμ
    T̃ = ntuple(k -> dot(γ[k], γw), 2)

    trB = B̃[1, 1] + B̃[2, 2]
    δn = γ[1] .* n₀[1] .+ γ[2] .* n₀[2]
    𝔽 = ntuple(
        k -> @.(γ[k] * δn - γ[k] * trB - (B̃[k, 1] * γ[1] + B̃[k, 2] * γ[2]) + T̃[k]), 2)

    Av = sum(basis_values(s, _unit_index(2, k)) *
             (st.μ .* (𝔻[k, 1] .* w[1] .+ 𝔻[k, 2] .* w[2] .- st.M .* 𝔽[k])) for k in 1:2)
    mass_factorization(s) \ Av
end

@doc raw"""
    metric_derivative(b::CollisionBracket, û)

The tensor ``\partial \mathbb{G}_{ij} / \partial \hat{u}_m``, analytic.

The state enters twice — through ``\hat{\phi} = \Lambda \hat{u}``, on which the operator
depends quadratically, and through ``M(x, u_h(x))``, in which it is bilinear — so each column
is one perturbed assembly and one sandwich, ``N`` of both. That is the same cost a
``\Lambda``-generated [`DoubleBracket`](@ref) pays, and for the same reason: an
``N \times N \times N`` tensor is not something a time loop should ask for, and a metriplectic
flow built on this bracket wants the directional derivative instead.

A prescribed ``\phi`` together with a state-independent mobility makes the bracket constant,
and the tensor is then returned as zeros without any assembly.
"""
function metric_derivative(b::CollisionBracket{T}, û::AbstractVector) where {T}
    s = b.space
    N = nbasis(s)
    st = _collision_state(b, û)

    dG = zeros(T, N, N, N)
    (b.h isa AbstractVector && all(iszero, st.Mu)) && return dG

    Φ = basis_values(s, (0, 0))
    zero_samples = zeros(T, length(st.μ))
    for m in 1:N
        δM = st.Mu .* Vector(Φ[m, :])
        δc = st.μ .* δM
        δγ = if b.h isa AbstractMatrix
            δφ̂ = Vector(b.h[:, m])
            (-field(s, δφ̂, (0, 1)), field(s, δφ̂, (1, 0)))
        else
            (zero_samples, zero_samples)
        end
        dG[m, :, :] = _mass_sandwich(s, _collision_operator_derivative(b, st, δγ, δc, δM))
    end
    return dG
end

@doc raw"""
    metric_directional(b::CollisionBracket, û, v)

The perturbed assemblies of [`metric_derivative`](@ref), each contracted against
``\mathbb{M}^{-1} v`` and solved once instead of sandwiched.

The operator derivative is dense here — the cross term is nonlocal — so this is ``N``
assemblies of size ``N^2`` and ``N + 1`` mass solves, against the same ``N`` assemblies plus
``N`` sandwiches. Of the four brackets this is the one where contracting first buys least: it
is the dense assemblies rather than the sandwiches that dominate at any size a
two-dimensional problem reaches, so the saving is a small constant factor and never an order.
Never slower, which is what the Newton iteration needs.
`scripts/verify_metriplectic_flow.jl` measures the ratio; it is a wall-clock number and is
reported there rather than pinned here, where it would go stale on the next machine.

The state enters through ``\hat{\phi} = \Lambda \hat{u}`` and through ``M(x, u_h)``, and the
same short-circuit applies: a prescribed ``\phi`` with a state-independent mobility makes the
bracket constant, and the derivative is zero without any assembly at all.
"""
function metric_directional(b::CollisionBracket{T}, û::AbstractVector,
        v::AbstractVector) where {T}
    s = b.space
    N = nbasis(s)
    st = _collision_state(b, û)

    D = zeros(T, N, N)
    (b.h isa AbstractVector && all(iszero, st.Mu)) && return D

    F = mass_factorization(s)
    w = F \ Vector(v)
    Φ = basis_values(s, (0, 0))
    zero_samples = zeros(T, length(st.μ))
    for m in 1:N
        δM = st.Mu .* Vector(Φ[m, :])
        δc = st.μ .* δM
        δγ = if b.h isa AbstractMatrix
            δφ̂ = Vector(b.h[:, m])
            (-field(s, δφ̂, (0, 1)), field(s, δφ̂, (1, 0)))
        else
            (zero_samples, zero_samples)
        end
        D[:, m] = F \ (_collision_operator_derivative(b, st, δγ, δc, δM) * w)
    end
    return D
end
