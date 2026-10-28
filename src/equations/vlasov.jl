@doc raw"""
    vlasov_galerkin_tensor(space::TensorSplineSpace{T, 2}) where {T}

The rank-3 tensor ``C[m, i, j] = c_{ij}^m`` of the direct Galerkin projection of the
two-dimensional Vlasov Lie-Poisson bracket

```math
\{F, G\}_{\mathrm{LP}}[f] = \int_\Omega f(x, v) \,
\left(
    \frac{\partial}{\partial x}\frac{\delta F}{\delta f} \,
    \frac{\partial}{\partial v}\frac{\delta G}{\delta f}
  - \frac{\partial}{\partial v}\frac{\delta F}{\delta f} \,
    \frac{\partial}{\partial x}\frac{\delta G}{\delta f}
\right) dx \, dv
```

onto `space`, with entries

```math
c_{ij}^m = \sum_{p, q = 1}^N (\mathbb{M}^{-1})_{ip} \, (\mathbb{M}^{-1})_{jq}
\int_\Omega \Phi_m(x, v) \, \{\Phi_p, \Phi_q\}_{(x, v)} \, dx \, dv .
```

The resulting linear Lie-Poisson matrix [`lie_poisson_matrix`](@ref)`(C, c)` is
antisymmetric for every coefficient vector `c`, but its [`structure_constant_residual`](@ref)
and [`jacobi_residual`](@ref) are ``O(1)`` because polynomial B-splines do not close under
the canonical phase-space Poisson bracket ``\{\cdot, \cdot\}_{(x, v)}``.
"""
function vlasov_galerkin_tensor(space::TensorSplineSpace{T, 2}) where {T}
    N = nbasis(space)
    Φ00 = Matrix(basis_values(space, (0, 0)))
    Φ10 = Matrix(basis_values(space, (1, 0)))
    Φ01 = Matrix(basis_values(space, (0, 1)))
    w = quadrature_weights(space)
    Minv = inverse_mass_matrix(space)

    Ψ10 = Minv * Φ10
    Ψ01 = Minv * Φ01

    C = zeros(T, N, N, N)
    wq = similar(w)
    for m in 1:N
        @inbounds for q in eachindex(w)
            wq[q] = w[q] * Φ00[m, q]
        end
        A = Ψ10 * Diagonal(wq) * Ψ01'
        @inbounds for j in 1:N, i in 1:N

            C[m, i, j] = A[i, j] - A[j, i]
        end
    end
    return C
end

@doc raw"""
    vlasov_weighted_poisson_matrix(space::TensorSplineSpace{T, 2}, f) where {T}

The ``N \times N`` antisymmetric matrix of the linearised Vlasov Lie-Poisson bracket frozen
at a background distribution `f` on `space`:

```math
\mathbb{J}_{ij}(f) = \sum_{p, q = 1}^N (\mathbb{M}^{-1})_{ip} \, (\mathbb{M}^{-1})_{jq}
\int_\Omega f(x, v) \, \{\Phi_p, \Phi_q\}_{(x, v)} \, dx \, dv .
```

`f` may be a function of a 2-tuple `(x, v)` or a vector of samples at
[`quadrature_nodes`](@ref)`(space)`.
"""
function vlasov_weighted_poisson_matrix(space::TensorSplineSpace{T, 2}, f) where {T}
    vlasov_weighted_poisson_matrix(space, map(f, quadrature_nodes(space)))
end

function vlasov_weighted_poisson_matrix(
        space::TensorSplineSpace{T, 2}, f::AbstractVector) where {T}
    w = quadrature_weights(space)
    length(f) == length(w) || throw(DimensionMismatch(
        "the distribution was sampled at $(length(f)) points but the quadrature grid has " *
        "$(length(w))"))
    Φ10 = Matrix(basis_values(space, (1, 0)))
    Φ01 = Matrix(basis_values(space, (0, 1)))
    Minv = inverse_mass_matrix(space)
    fw = f .* w
    W = Φ10 * Diagonal(fw) * Φ01' - Φ01 * Diagonal(fw) * Φ10'
    J = Minv * W * Minv'
    return (J - J') / 2
end

@doc raw"""
    velocity_parity_projectors(space::TensorSplineSpace{T, 2}; orthonormal::Bool = false) where {T}

The embedding matrices `(U_plus, U_minus)` of the ``v``-even and ``v``-odd subspaces of a
two-dimensional tensor-product spline space under velocity reflection
``\mathcal{T}_v(x, v) = (x, L_v - v)``, which reverses the velocity B-spline index
``j \mapsto n_v + 1 - j``.

When `orthonormal = false`, the columns of `U_plus` and `U_minus` are Euclidean-orthonormal
(``U_\pm^T U_\pm = \mathbb{I}``, ``U_+^T U_- = \mathbb{O}``). When `orthonormal = true`,
they are ``\mathbb{M}``-orthonormalised via the symmetric Cholesky factorisation of the
mass matrix (``U_\pm^T \mathbb{M} U_\pm = \mathbb{I}``, ``U_+^T \mathbb{M} U_- = \mathbb{O}``),
forming the parity-adapted B-spline Stiefel frame.
"""
function velocity_parity_projectors(
        space::TensorSplineSpace{T, 2}; orthonormal::Bool = false) where {T}
    nx, nv = size(space)
    m_minus = div(nv, 2)
    m_plus = nv - m_minus

    E_plus = zeros(T, nv, m_plus)
    E_minus = zeros(T, nv, m_minus)
    inv_sqrt2 = inv(sqrt(T(2)))
    for k in 1:m_minus
        rk = nv + 1 - k
        E_plus[k, k] = inv_sqrt2
        E_plus[rk, k] = inv_sqrt2
        E_minus[k, k] = inv_sqrt2
        E_minus[rk, k] = -inv_sqrt2
    end
    if isodd(nv)
        mid = m_minus + 1
        E_plus[mid, m_plus] = one(T)
    end

    Ix = Matrix{T}(I, nx, nx)
    U_plus = kron(E_plus, Ix)
    U_minus = kron(E_minus, Ix)

    if orthonormal
        M = Matrix(mass_matrix(space))
        M_plus = Symmetric(U_plus' * M * U_plus)
        M_minus = Symmetric(U_minus' * M * U_minus)
        U_plus = U_plus / cholesky(M_plus).U
        if m_minus > 0
            U_minus = U_minus / cholesky(M_minus).U
        end
    end
    return U_plus, U_minus
end

@doc raw"""
    parity_vlasov_blocks(space::TensorSplineSpace{T, 2}, f0; orthonormal::Bool = false) where {T}

The four parity blocks `(J_pp, J_pm, J_mp, J_mm)` of the linearised Vlasov Lie-Poisson
operator at background `f0` in the ``(v\text{-even}, v\text{-odd})`` decomposition from
[`velocity_parity_projectors`](@ref).

For any velocity-even background ``f_0(x, L_v - v) = f_0(x, v)`` on a velocity basis with
reflection symmetry (such as `Free()` or `Dirichlet()` on the velocity axis), the even and
odd subspaces are Lagrangian (maximally isotropic): `J_pp` and `J_mm` vanish to machine
precision, and `J_mp == -J_pm'`.
"""
function parity_vlasov_blocks(
        space::TensorSplineSpace{T, 2}, f0; orthonormal::Bool = false) where {T}
    parity_vlasov_blocks(
        space, map(f0, quadrature_nodes(space)); orthonormal = orthonormal)
end

function parity_vlasov_blocks(space::TensorSplineSpace{T, 2}, f0::AbstractVector;
        orthonormal::Bool = false) where {T}
    U_plus, U_minus = velocity_parity_projectors(space; orthonormal = orthonormal)
    if orthonormal
        w = quadrature_weights(space)
        length(f0) == length(w) || throw(DimensionMismatch(
            "the distribution was sampled at $(length(f0)) points but the quadrature grid has " *
            "$(length(w))"))
        Φ10 = Matrix(basis_values(space, (1, 0)))
        Φ01 = Matrix(basis_values(space, (0, 1)))
        fw = f0 .* w
        J = Φ10 * Diagonal(fw) * Φ01' - Φ01 * Diagonal(fw) * Φ10'
        J = (J - J') / 2
    else
        J = vlasov_weighted_poisson_matrix(space, f0)
    end
    J_pp = (U_plus' * J * U_plus - (U_plus' * J * U_plus)') / 2
    J_pm = U_plus' * J * U_minus
    J_mp = -J_pm'
    J_mm = (U_minus' * J * U_minus - (U_minus' * J * U_minus)') / 2
    return J_pp, J_pm, J_mp, J_mm
end

@doc raw"""
    parity_vlasov_bracket(A::AbstractMatrix{T}) where {T}
    parity_vlasov_bracket(space::TensorSplineSpace{T, 2}, f0; orthonormal::Bool = false) where {T}

The constant skew-symmetric block-off-diagonal Poisson bracket

```math
\mathbb{J}_{\mathrm{par}} =
\begin{pmatrix}
\mathbb{O}_{N_+ \times N_+} & \mathbb{A} \\
-\mathbb{A}^T & \mathbb{O}_{N_- \times N_-}
\end{pmatrix}
```

on the time-reversal parity-split phase space
``(Q^{(0)}, P^{(0)}) \in \mathbb{R}^{N_+ + N_-}``, wrapped as a [`ConstantBracket`](@ref).
Because ``\mathbb{J}_{\mathrm{par}}`` is independent of ``(Q^{(0)}, P^{(0)})``, its
derivative tensor vanishes identically and it satisfies the Jacobi identity at every
resolution and for every coupling matrix ``\mathbb{A}``.
"""
function parity_vlasov_bracket(A::AbstractMatrix{T}) where {T}
    np, nm = size(A)
    J = zeros(T, np + nm, np + nm)
    J[1:np, (np + 1):(np + nm)] .= A
    J[(np + 1):(np + nm), 1:np] .= -transpose(A)
    return ConstantBracket(J)
end

function parity_vlasov_bracket(
        space::TensorSplineSpace{T, 2}, f0; orthonormal::Bool = false) where {T}
    _, J_pm, _, _ = parity_vlasov_blocks(space, f0; orthonormal = orthonormal)
    return parity_vlasov_bracket(J_pm)
end

@doc raw"""
    dlra_parity_vlasov_bracket(A::AbstractMatrix, Ux::AbstractMatrix,
                               Uv_plus::AbstractMatrix, Uv_minus::AbstractMatrix)

The ``O(r) \times O(r)`` gauge-invariant dynamic-low-rank parity Vlasov bracket obtained by
projecting the full parity coupling matrix `A` (of size ``(n_x m_+) \times (n_x m_-)``) onto
low-rank spatial and velocity Stiefel frames `Ux`, `Uv_plus` and `Uv_minus`:

```math
\mathbb{A}_{\mathrm{dlra}} = (U_{v,+}^T \otimes U_x^T) \, \mathbb{A} \, (U_{v,-} \otimes U_x) .
```

Returns [`parity_vlasov_bracket`](@ref)`(A_dlra)` as an exact [`ConstantBracket`](@ref).
"""
function dlra_parity_vlasov_bracket(A::AbstractMatrix, Ux::AbstractMatrix,
        Uv_plus::AbstractMatrix, Uv_minus::AbstractMatrix)
    P_plus = kron(Uv_plus, Ux)
    P_minus = kron(Uv_minus, Ux)
    A_dlra = P_plus' * A * P_minus
    return parity_vlasov_bracket(A_dlra)
end

@doc raw"""
    canonical_vlasov_bracket(N::Integer, ::Type{T} = Float64) where {T}

The canonical symplectic [`ConstantBracket`](@ref) on ``\mathbb{R}^{2N}`` with Poisson matrix

```math
\mathbb{J}_{2N} =
\begin{pmatrix}
\mathbb{O}_N & \mathbb{I}_N \\
-\mathbb{I}_N & \mathbb{O}_N
\end{pmatrix} .
```
"""
function canonical_vlasov_bracket(N::Integer, ::Type{T} = Float64) where {T}
    N >= 1 || throw(ArgumentError("N must be at least 1, got $N"))
    J = zeros(T, 2N, 2N)
    for i in 1:N
        J[i, N + i] = one(T)
        J[N + i, i] = -one(T)
    end
    return ConstantBracket(J)
end

@doc raw"""
    clebsch_vlasov_tensor(space::TensorSplineSpace{T, 2}) where {T}

The rank-3 discrete Clebsch bilinear coupling tensor ``B[m, i, j] = \mathcal{B}_{m, ij}``
for ``f_h = f_{0, h} + \Pi_h\{\alpha_h, \beta_h\}_{(x, v)}`` on `space`:

```math
\mathcal{B}_{m, ij} = \sum_{p = 1}^N (\mathbb{M}^{-1})_{mp}
\int_\Omega \Phi_p(x, v) \, \{\Phi_i, \Phi_j\}_{(x, v)} \, dx \, dv .
```

For each mode `m`, the slice `B[m, :, :]` is skew-symmetric, and because
``\sum_{p = 1}^N \Phi_p(x, v) \equiv 1`` on a B-spline partition of unity, the basis
integrals ``b_m = \int_\Omega \Phi_m \, dx \, dv`` annihilate the Clebsch perturbation
identically on a periodic domain: ``\sum_m b_m \mathcal{B}_{m, ij} = 0``.
"""
function clebsch_vlasov_tensor(space::TensorSplineSpace{T, 2}) where {T}
    N = nbasis(space)
    Φ00 = Matrix(basis_values(space, (0, 0)))
    Φ10 = Matrix(basis_values(space, (1, 0)))
    Φ01 = Matrix(basis_values(space, (0, 1)))
    w = quadrature_weights(space)
    Minv = inverse_mass_matrix(space)

    Ψ00 = Minv * Φ00
    B = zeros(T, N, N, N)
    wq = similar(w)
    for m in 1:N
        @inbounds for q in eachindex(w)
            wq[q] = w[q] * Ψ00[m, q]
        end
        A = Φ10 * Diagonal(wq) * Φ01'
        @inbounds for j in 1:N, i in 1:N

            B[m, i, j] = A[i, j] - A[j, i]
        end
    end
    return B
end

@doc raw"""
    clebsch_vlasov_map(B::AbstractArray{<:Any, 3}, Q::AbstractVector, P::AbstractVector,
                       c0::AbstractVector = zeros(promote_type(eltype(B), eltype(Q), eltype(P)), size(B, 1)))

Evaluates the discrete Clebsch momentum map

```math
c_m(Q, P) = c_{0, m} + \sum_{i, j = 1}^N B[m, i, j] \, Q_i P_j ,
```

mapping canonical Clebsch coordinates ``(Q, P) \in \mathbb{R}^{2N}`` to Eulerian basis
coefficients ``c \in \mathbb{R}^M``.
"""
function clebsch_vlasov_map(B::AbstractArray{<:Any, 3}, Q::AbstractVector,
        P::AbstractVector,
        c0::AbstractVector = zeros(
            promote_type(eltype(B), eltype(Q), eltype(P)), size(B, 1)))
    M, N1, N2 = size(B)
    (length(Q) == N1 && length(P) == N2 && length(c0) == M) || throw(DimensionMismatch(
        "Clebsch tensor of size $(size(B)) is incompatible with lengths ($(length(Q)), $(length(P)), $(length(c0)))"))
    RT = promote_type(eltype(B), eltype(Q), eltype(P), eltype(c0))
    c = Vector{RT}(undef, M)
    for m in 1:M
        Bm = @view B[m, :, :]
        c[m] = c0[m] + dot(Q, Bm * P)
    end
    return c
end

@doc raw"""
    clebsch_pushforward_matrix(B::AbstractArray{<:Any, 3}, Q::AbstractVector, P::AbstractVector)

The ``M \times M`` pushforward bivector

```math
\Pi_{mn}(Q, P) = \{c_m, c_n\}_{\mathrm{can}}(Q, P) =
(\nabla_Q c_m)^T (\nabla_P c_n) - (\nabla_P c_m)^T (\nabla_Q c_n)
```

induced on the Eulerian coefficient space ``\mathbb{R}^M`` by the canonical bracket
[`canonical_vlasov_bracket`](@ref) under [`clebsch_vlasov_map`](@ref).

By the discrete Clebsch non-submersion theorem, when each slice ``B_m = B[m, :, :]`` is
skew-symmetric, ``\Pi_{mn}(Q, P) = -Q^T [B_m, B_n] P``, so ``\Pi(Q, P)`` descends to a
linear Lie-Poisson bracket on ``c`` if and only if the slices ``B[m, :, :]`` close under
the matrix commutator.
"""
function clebsch_pushforward_matrix(
        B::AbstractArray{<:Any, 3}, Q::AbstractVector, P::AbstractVector)
    M, N1, N2 = size(B)
    (length(Q) == N1 && length(P) == N2) || throw(DimensionMismatch(
        "Clebsch tensor of size $(size(B)) is incompatible with (Q, P) of lengths ($(length(Q)), $(length(P)))"))
    RT = promote_type(eltype(B), eltype(Q), eltype(P))
    G_Q = Matrix{RT}(undef, N1, M)
    G_P = Matrix{RT}(undef, N2, M)
    for m in 1:M
        Bm = @view B[m, :, :]
        G_Q[:, m] = Bm * P
        G_P[:, m] = Bm' * Q
    end
    return G_Q' * G_P - G_P' * G_Q
end

@doc raw"""
    un_hermitian_basis(N::Integer, ::Type{T} = Float64) where {T}

An orthonormal basis ``\{E_a\}_{a = 1}^{N^2}`` of ``N \times N`` Hermitian matrices with
respect to the trace inner product ``\operatorname{tr}(E_a E_b) = \delta_{ab}``: the ``N``
diagonal projectors ``e_j e_j^T``, the ``N(N - 1)/2`` real symmetric off-diagonal matrices
``(e_j e_k^T + e_k e_j^T)/\sqrt{2}``, and the ``N(N - 1)/2`` imaginary skew-symmetric
matrices ``i(e_j e_k^T - e_k e_j^T)/\sqrt{2}``.
"""
function un_hermitian_basis(N::Integer, ::Type{T} = Float64) where {T}
    N >= 1 || throw(ArgumentError("N must be at least 1, got $N"))
    basis = Vector{Matrix{Complex{T}}}(undef, N * N)
    idx = 0
    for j in 1:N
        idx += 1
        E = zeros(Complex{T}, N, N)
        E[j, j] = one(T)
        basis[idx] = E
    end
    inv_sqrt2 = inv(sqrt(T(2)))
    for j in 1:N, k in (j + 1):N

        idx += 1
        Es = zeros(Complex{T}, N, N)
        Es[j, k] = inv_sqrt2
        Es[k, j] = inv_sqrt2
        basis[idx] = Es

        idx += 1
        Ea = zeros(Complex{T}, N, N)
        Ea[j, k] = Complex{T}(zero(T), inv_sqrt2)
        Ea[k, j] = Complex{T}(zero(T), -inv_sqrt2)
        basis[idx] = Ea
    end
    return basis
end

@doc raw"""
    un_structure_constants(N::Integer, ::Type{T} = Float64; ħ::Real = one(T)) where {T}

The real ``N^2 \times N^2 \times N^2`` structure-constant tensor

```math
C[m, a, b] = c_{ab}^m = \operatorname{tr}\!\left(E_m \, \frac{1}{i\hbar}[E_a, E_b]\right)
```

of the finite-mode von Neumann-Poisson bracket on ``\mathfrak{u}(N)^*`` in the trace-orthonormal
Hermitian basis [`un_hermitian_basis`](@ref). Because ``\mathfrak{u}(N)`` is a compact Lie
algebra and ``\operatorname{tr}(E_a E_b) = \delta_{ab}`` is ad-invariant, ``c_{ab}^m`` is
completely antisymmetric in ``(m, a, b)`` and satisfies [`structure_constant_residual`](@ref)
to machine precision at every ``N``.
"""
function un_structure_constants(
        N::Integer, ::Type{T} = Float64; ħ::Real = one(T)) where {T}
    E = un_hermitian_basis(N, T)
    M = length(E)
    C = zeros(T, M, M, M)
    inv_ħ = inv(T(ħ))
    for b in 1:M, a in 1:(b - 1)

        comm = (E[a] * E[b] - E[b] * E[a]) * Complex{T}(zero(T), -inv_ħ)
        for m in 1:M
            val = real(tr(E[m] * comm))
            C[m, a, b] = val
            C[m, b, a] = -val
        end
    end
    return C
end

@doc raw"""
    weyl_vlasov_matrix(modes::AbstractDict{Tuple{Int, Int}, <:Number}, K::Integer, ħ::Real)

The ``(2K + 1) \times (2K + 1)`` compressed Weyl quantisation matrix ``T_N(a)`` of a
polynomial-in-momentum Vlasov symbol

```math
a(x, v) = \sum_{(n, r)} a_{n, r} \, e^{i n x} \, v^r
```

on ``S^1 \times \mathbb{R}`` over the Fourier mode window ``k, l \in -K:K``, with matrix
entries

```math
T_N(a)_{k, l} = \sum_{(n, r): \, k - l = n} a_{n, r} \,
\left(\hbar \, \frac{k + l}{2}\right)^r .
```
"""
function weyl_vlasov_matrix(
        modes::AbstractDict{Tuple{Int, Int}, <:Number}, K::Integer, ħ::Real)
    K >= 0 || throw(ArgumentError("K must be non-negative, got $K"))
    N = 2K + 1
    VT = promote_type(valtype(modes), typeof(ħ))
    CT = Complex{real(float(VT))}
    A = zeros(CT, N, N)
    for ((n, r), coeff) in modes
        for l in (-K):K
            k = l + n
            (-K <= k <= K) || continue
            pmid = (ħ * (k + l)) / 2
            A[k + K + 1, l + K + 1] += CT(coeff) * (pmid^r)
        end
    end
    return A
end

function _hermitian_eigen(W::AbstractMatrix{Complex{T}}) where {T <: LinearAlgebra.BlasReal}
    eig = eigen(Hermitian(W))
    return eig.values, eig.vectors
end

function _hermitian_eigen(W::AbstractMatrix{Complex{T}}) where {T <: Real}
    N = size(W, 1)
    A = Matrix{Complex{T}}(W)
    for j in 1:N, i in 1:N

        A[i, j] = (W[i, j] + conj(W[j, i])) / 2
    end
    U = Matrix{Complex{T}}(I, N, N)
    tol = eps(T) * T(10)
    for _ in 1:60
        off = zero(T)
        for q in 2:N, p in 1:(q - 1)

            off = max(off, abs(A[p, q]))
        end
        off <= tol && break
        for q in 2:N, p in 1:(q - 1)

            apq = A[p, q]
            r = abs(apq)
            r <= tol && continue
            phase = apq / r
            tau = real(A[q, q] - A[p, p]) / (2 * r)
            t = tau >= 0 ? one(T) / (tau + sqrt(one(T) + tau * tau)) :
                -one(T) / (-tau + sqrt(one(T) + tau * tau))
            c = one(T) / sqrt(one(T) + t * t)
            s = t * c
            for k in 1:N
                ikp = A[k, p]
                ikq = A[k, q]
                A[k, p] = c * ikp - s * conj(phase) * ikq
                A[k, q] = s * phase * ikp + c * ikq
            end
            for k in 1:N
                apk = A[p, k]
                aqk = A[q, k]
                A[p, k] = c * apk - s * phase * aqk
                A[q, k] = s * conj(phase) * apk + c * aqk
            end
            A[p, q] = zero(Complex{T})
            A[q, p] = zero(Complex{T})
            for k in 1:N
                ukp = U[k, p]
                ukq = U[k, q]
                U[k, p] = c * ukp - s * conj(phase) * ukq
                U[k, q] = s * phase * ukp + c * ukq
            end
        end
    end
    λ = [real(A[i, i]) for i in 1:N]
    perm = sortperm(λ)
    return λ[perm], U[:, perm]
end

struct _ComplexStepSecondDeriv{DF}
    dφ::DF
end

function (cs::_ComplexStepSecondDeriv)(x::T) where {T <: Real}
    h = eps(T)
    return imag(cs.dφ(complex(x, h))) / h
end

@doc raw"""
    spectral_pushforward_bracket(φ, dφ, w::AbstractVector{T}; ħ::Real = one(T)) where {T}
    spectral_pushforward_bracket(φ, dφ, d2φ, w::AbstractVector{T}; ħ::Real = one(T)) where {T}

The ``N^2 \times N^2`` Poisson matrix `J` and derivative tensor `dJ` (with
``dJ[m, a, b] = \partial J_{ab} / \partial v_m``) in ``v``-coordinates of the pushforward
of the ``\mathfrak{u}(N)^*`` Lie-Poisson bracket along the monotone spectral diffeomorphism
``V = \varphi(W)``, evaluated analytically at ``W = \sum_{a = 1}^{N^2} w_a E_a`` in the
basis [`un_hermitian_basis`](@ref) via first- and second-order Daleckii-Krein divided
differences ``\mathcal{L}_\varphi(\lambda_j, \lambda_k)`` and
``\mathcal{L}_\varphi^{[2]}(\lambda_i, \lambda_j, \lambda_\ell)``.

For ``\varphi(\lambda) = \tfrac{1}{2}e^{2\lambda}`` (corresponding to the quadratic
structure function ``c(w) = w^2``), ``V \succ 0`` is strictly positive-definite and the
Fréchet derivative ``D\varphi(W)`` acts on eigenbasis entries by the logarithmic mean
``(v_j - v_k)/(\log v_j - \log v_k)``, yielding the Bogoliubov-Kubo-Mori logarithmic-mean
Poisson bracket on ``\mathfrak{u}(N)^*`` which satisfies the Jacobi identity to machine
precision in both `Float64` and `BigFloat`.
"""
function spectral_pushforward_bracket(
        φ::F, dφ::DF, w::AbstractVector{T};
        ħ::Real = one(T)) where {F, DF, T <: Real}
    return spectral_pushforward_bracket(φ, dφ, _ComplexStepSecondDeriv(dφ), w; ħ = ħ)
end

function spectral_pushforward_bracket(
        φ::F, dφ::DF, d2φ::D2F, w::AbstractVector{T};
        ħ::Real = one(T)) where {F, DF, D2F, T <: Real}
    M = length(w)
    N = isqrt(M)
    N * N == M || throw(DimensionMismatch("length(w) = $M must be a perfect square N^2"))
    E = un_hermitian_basis(N, T)
    C = un_structure_constants(N, T; ħ = ħ)

    W = zeros(Complex{T}, N, N)
    for a in 1:M
        W .+= w[a] .* E[a]
    end
    λ, U = _hermitian_eigen(W)
    tol = sqrt(eps(T))
    L1 = Matrix{T}(undef, N, N)
    for k in 1:N, j in 1:N

        if abs(λ[j] - λ[k]) < tol
            L1[j, k] = dφ((λ[j] + λ[k]) / 2)
        else
            L1[j, k] = (φ(λ[j]) - φ(λ[k])) / (λ[j] - λ[k])
        end
    end
    L2 = Array{T, 3}(undef, N, N, N)
    for l in 1:N, j in 1:N, i in 1:N
        if abs(λ[i] - λ[j]) < tol && abs(λ[j] - λ[l]) < tol
            L2[i, j, l] = d2φ((λ[i] + λ[j] + λ[l]) / 3) / 2
        elseif abs(λ[j] - λ[l]) < tol
            λbar = (λ[j] + λ[l]) / 2
            L2[i, j, l] = (L1[i, j] - dφ(λbar)) / (λ[i] - λbar)
        else
            L2[i, j, l] = (L1[i, j] - L1[i, l]) / (λ[j] - λ[l])
        end
    end

    E_hat = [U' * E[a] * U for a in 1:M]
    Jac = zeros(T, M, M)
    for b in 1:M
        D_Eb = L1 .* E_hat[b]
        for a in 1:M
            Jac[a, b] = real(tr(E_hat[a]' * D_Eb))
        end
    end

    # Second Fréchet derivative dJac[k, a, b] = ∂Jac[a, b] / ∂w_k via L2 divided differences
    dJac = zeros(T, M, M, M)
    D2_bk = Matrix{Complex{T}}(undef, N, N)
    for k in 1:M, b in k:M

        Ek = E_hat[k]
        Eb = E_hat[b]
        for l in 1:N, i in 1:N

            s = zero(Complex{T})
            for j in 1:N
                s += L2[i, j, l] * (Eb[i, j] * Ek[j, l] + Ek[i, j] * Eb[j, l])
            end
            D2_bk[i, l] = s
        end
        for a in 1:M
            val = real(tr(E_hat[a]' * D2_bk))
            dJac[k, a, b] = val
            dJac[b, a, k] = val
        end
    end

    J_w = lie_poisson_matrix(C, w)
    J_v = Jac * J_w * Jac'
    dJ_w = zeros(T, M, M, M)
    for k in 1:M
        dJac_k = @view dJac[k, :, :]
        Ck = @view C[k, :, :]
        dJ_w[k, :, :] .= dJac_k * J_w * Jac' .+ Jac * Ck * Jac' .+ Jac * J_w * dJac_k'
    end
    Jac_inv = inv(Jac)
    dJ_v = zeros(T, M, M, M)
    for m in 1:M, k in 1:M

        dJ_v[m, :, :] .+= Jac_inv[k, m] .* @view(dJ_w[k, :, :])
    end
    return J_v, dJ_v
end

@doc raw"""
    vlasov_moment_flat_bracket(Ka::AbstractMatrix, ρ::AbstractVector, m::AbstractVector;
                               Kb::AbstractMatrix = zeros(promote_type(eltype(Ka), eltype(ρ), eltype(m)), size(Ka)))

The ``2N \times 2N`` discrete Poisson matrix `J` and derivative tensor `dJ`
(``dJ[k, i, j] = \partial J_{ij} / \partial z_k``) of the pointwise-flat semidirect-product
Vlasov moment / waterbag bracket on ``(\rho, m)`` (density ``\rho_i = \int f \, dv`` and
momentum density ``m_i = \int v f \, dv``), obtained by pushing forward the constant bracket

```math
\mathbb{K}_2 =
\begin{pmatrix}
\mathbb{O}_N & \mathbb{K}_a \\
-\mathbb{K}_a^T & \mathbb{K}_b
\end{pmatrix}
```

from the flat velocity chart ``(\rho_i, u_i = m_i / \rho_i)`` via
``D\Phi_N = \begin{psmallmatrix} \mathbb{I} & \mathbb{O} \\ \mathrm{diag}(u) & \mathrm{diag}(\rho) \end{psmallmatrix}``:

```math
\mathbb{J}(\rho, m) = D\Phi_N \, \mathbb{K}_2 \, D\Phi_N^T =
\begin{pmatrix}
\mathbb{O}_N & \mathbb{K}_a \mathrm{diag}(\rho) \\
-\mathrm{diag}(\rho) \mathbb{K}_a^T &
\mathrm{diag}(u) \mathbb{K}_a \mathrm{diag}(\rho)
- \mathrm{diag}(\rho) \mathbb{K}_a^T \mathrm{diag}(u)
+ \mathrm{diag}(\rho) \mathbb{K}_b \mathrm{diag}(\rho)
\end{pmatrix} .
```

Satisfies the Jacobi identity identically for **every** matrix ``\mathbb{K}_a`` and every
antisymmetric matrix ``\mathbb{K}_b = -\mathbb{K}_b^T`` on both periodic and bounded domains.
"""
function vlasov_moment_flat_bracket(
        Ka::AbstractMatrix, ρ::AbstractVector, m::AbstractVector;
        Kb::AbstractMatrix = zeros(
            promote_type(eltype(Ka), eltype(ρ), eltype(m)), size(Ka)))
    N = length(ρ)
    (size(Ka) == (N, N) && size(Kb) == (N, N) && length(m) == N) ||
        throw(DimensionMismatch(
            "expected N×N matrices and length-N vectors, got size(Ka)=$(size(Ka)), size(Kb)=$(size(Kb)), length(ρ)=$N, length(m)=$(length(m))"))
    T = promote_type(eltype(Ka), eltype(Kb), eltype(ρ), eltype(m))
    u = Vector{T}(undef, N)
    for i in 1:N
        u[i] = m[i] / ρ[i]
    end
    J = zeros(T, 2N, 2N)
    dJ = zeros(T, 2N, 2N, 2N)
    for j in 1:N, i in 1:N

        J[i, N + j] = Ka[i, j] * ρ[j]
        J[N + i, j] = -ρ[i] * Ka[j, i]
        J[N + i, N + j] = u[i] * Ka[i, j] * ρ[j] - ρ[i] * Ka[j, i] * u[j] +
                          ρ[i] * Kb[i, j] * ρ[j]

        dJ[j, i, N + j] += Ka[i, j]
        dJ[i, N + i, j] -= Ka[j, i]

        # ∂/∂ρ_i and ∂/∂ρ_j of J_{N+i, N+j} at fixed m (since u_k = m_k / ρ_k)
        dJ[i, N + i, N + j] += -(m[i] / (ρ[i]^2)) * Ka[i, j] * ρ[j] - Ka[j, i] * u[j] +
                               Kb[i, j] * ρ[j]
        dJ[j, N + i, N + j] += u[i] * Ka[i, j] + ρ[i] * Ka[j, i] * (m[j] / (ρ[j]^2)) +
                               ρ[i] * Kb[i, j]

        # ∂/∂m_i and ∂/∂m_j of J_{N+i, N+j} at fixed ρ
        dJ[N + i, N + i, N + j] += (one(T) / ρ[i]) * Ka[i, j] * ρ[j]
        dJ[N + j, N + i, N + j] -= ρ[i] * Ka[j, i] * (one(T) / ρ[j])
    end
    return J, dJ
end

@doc raw"""
    vlasov_moment_direct_bracket(K::AbstractMatrix, ρ::AbstractVector, m::AbstractVector)

The ``2N \times 2N`` matrix `J` and derivative tensor `dJ`
(``dJ[k, i, j] = \partial J_{ij} / \partial z_k``) of the naive direct-collocation
discretisation of the semidirect-product Vlasov moment bracket:

```math
\mathbb{J}_{\mathrm{dir}}(\rho, m) =
\begin{pmatrix}
\mathbb{O}_N & \mathbb{K}\mathrm{diag}(\rho) \\
-\mathrm{diag}(\rho)\mathbb{K}^T & \mathrm{diag}(m)\mathbb{K} - \mathbb{K}^T\mathrm{diag}(m)
\end{pmatrix} .
```

Unlike [`vlasov_moment_flat_bracket`](@ref), this linear-in-state discretisation violates
the Jacobi identity at ``O(1)`` because the ``(\rho, m)``-block and ``(m, m)``-block fail
to satisfy the Leibniz product rule on a discrete stencil.
"""
function vlasov_moment_direct_bracket(
        K::AbstractMatrix, ρ::AbstractVector, m::AbstractVector)
    N = length(ρ)
    (size(K) == (N, N) && length(m) == N) || throw(DimensionMismatch(
        "expected N×N matrix and length-N vectors, got size(K)=$(size(K)), length(ρ)=$N, length(m)=$(length(m))"))
    T = promote_type(eltype(K), eltype(ρ), eltype(m))
    J = zeros(T, 2N, 2N)
    dJ = zeros(T, 2N, 2N, 2N)
    for j in 1:N, i in 1:N

        J[i, N + j] = K[i, j] * ρ[j]
        J[N + i, j] = -ρ[i] * K[j, i]
        J[N + i, N + j] = m[i] * K[i, j] - K[j, i] * m[j]

        dJ[j, i, N + j] += K[i, j]
        dJ[i, N + i, j] -= K[j, i]
        dJ[N + i, N + i, N + j] += K[i, j]
        dJ[N + j, N + i, N + j] -= K[j, i]
    end
    return J, dJ
end
