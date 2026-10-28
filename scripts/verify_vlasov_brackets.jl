#!/usr/bin/env julia
#
# Finite-dimensional Poisson brackets for the Vlasov equation: obstructions and four
# exact non-particle constructions.
#
#     julia --project=scripts scripts/verify_vlasov_brackets.jl
#
# Companion verification script to `discrete-vlasov-brackets.tex`. Checks every positive
# theorem alongside a negative control that breaks one hypothesis:
#
#   1. Direct Eulerian discretisations (B-spline Galerkin, 9-point Arakawa, truncated
#      Fourier) violate the Jacobi identity at O(1); pointwise variable transformations
#      cannot flatten a 2D canonical bracket; Dirac reduction preserves the O(1) defect.
#   2. Time-reversal parity splitting (f = f0 + δf+ + δf-) around any v-even background f0
#      makes the v-even and v-odd subspaces Lagrangian (J++ = J-- = 0), yielding a constant
#      block-off-diagonal Poisson bracket on spline, grid and O(r)×O(r) DLRA Stiefel frames,
#      plus exact canonical pullback under nonlinear symplectic shears and affine reference
#      shifts. Control: a drifted (v-skew) background breaks isotropy (J++, J-- != 0).
#   3. Canonical Clebsch potentials f_h = f_{0,h} + Π_h{α_h, β_h} carry the exact canonical
#      bracket J_{2N} on (Q, P) ∈ R^{2N} and conserve total mass on T^2, while the Discrete
#      Clebsch Non-Submersion Theorem shows that the pushforward Π(Q, P) = -Q^T [B_m, B_n] P
#      descends to a Lie-Poisson bracket on c iff {B_m} closes under the matrix commutator.
#   4. Finite-mode von Neumann-Poisson quantisation on u(N)* satisfies Jacobi identically,
#      is exact on the inner block for degree <= 1 in v, has exact Wigner O(ħ^2) dispersion
#      at degree 2, and admits both the positive-definite logarithmic-mean (BKM) spectral
#      pushforward and exact Dirac reduction. Control: naive tr(W^2 [F_W, G_W]) fails Jacobi.
#   5. Pointwise-flat semidirect-product Vlasov moment / waterbag brackets on (ρ, m) satisfy
#      Jacobi identically in exact Rational{BigInt} arithmetic for both periodic and
#      wall-bounded operators Ka. Control: direct collocation fails Jacobi at O(1).

using GeometricBrackets
using GeometricBrackets: poisson_derivative
using LinearAlgebra
using Printf
using Random
using SimpleSplines: Periodic, Free, Dirichlet

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

# =============================================================================
# 1. Direct Eulerian discretisations and structural obstructions
# =============================================================================
header("1. Direct Eulerian Vlasov projections and structural obstructions")

for (label, bc) in (("T^2 periodic", (Periodic(), Periodic())),
    ("S^1 × [-V, V] clamped", (Periodic(), Free())))
    s = TensorSplineSpace((3, 3), 2, bc)
    C = vlasov_galerkin_tensor(s)
    rng = MersenneTwister(42)
    c = randn(rng, nbasis(s))
    J = lie_poisson_matrix(C, c)
    sc_res = structure_constant_residual(C)
    jac_res = jacobi_residual(J, C)
    check("Galerkin Vlasov ($label): C is antisymmetric",
        is_antisymmetric_c(C),
        @sprintf("max|C + C^T| = %.2e", maximum(abs, C .+ permutedims(C, (1, 3, 2)))))
    check("Galerkin Vlasov ($label): violates Jacobi at O(1) (control)",
        sc_res > 0.05 && jac_res > 0.05,
        @sprintf("sc_res = %.3e, jac_res = %.3e", sc_res, jac_res))
end

for n in (5, 7)
    arak = Arakawa(n, n, 2π / n, 2π / n)
    u_grid = randn(MersenneTwister(n), n * n)
    C_arak = poisson_derivative(arak, u_grid)
    J_arak = poisson_matrix(arak, u_grid)
    sc_arak = structure_constant_residual(C_arak)
    jac_arak = jacobi_residual(J_arak, C_arak)
    check("Arakawa $(n)×$(n) grid: structure_constant_residual == 0.5 (control)",
        isapprox(sc_arak, 0.5; atol = 1e-12) && jac_arak > 0.1,
        @sprintf("sc_res = %.6f, jac_res = %.3e", sc_arak, jac_arak))
end

let N = 5
    C_trunc = torus_truncation(Float64, 1; q = 1).C
    _, C_zeit = sine_algebra(N)
    sc_trunc = structure_constant_residual(C_trunc)
    sc_zeit = structure_constant_residual(C_zeit)
    check("Truncated Fourier on T^2 (K=1): fails closure at O(1) (control)",
        sc_trunc > 0.1, @sprintf("sc_res = %.3e", sc_trunc))
    check("Zeitlin sine algebra on su($N)*: closes to machine precision",
        sc_zeit < 1e-13, @sprintf("sc_res = %.2e", sc_zeit))
end

# 2D pointwise flattening obstruction: {h F, h G}_{(x,v)} != h^2 {F, G}_{(x,v)}
let g = spectral_grid(32)
    h_f = sample(g, (x, y) -> 2.0 + 0.5 * cos(x) * sin(y))
    F_f = sample(g, (x, y) -> sin(x) + 0.3 * cos(2y))
    G_f = sample(g, (x, y) -> cos(x - y) - 0.2 * sin(2x))
    # 1D Wronskian [h F, h G]_x = h F ∂_x(h G) - h G ∂_x(h F) == h^2 (F ∂_x G - G ∂_x F)
    w1_lhs = (h_f .* F_f) .* ∂x(g, h_f .* G_f) .- (h_f .* G_f) .* ∂x(g, h_f .* F_f)
    w1_rhs = (h_f .^ 2) .* (F_f .* ∂x(g, G_f) .- G_f .* ∂x(g, F_f))
    err_1d = maximum(abs, w1_lhs .- w1_rhs)
    # 2D canonical bracket {h F, h G}_{(x,y)} vs h^2 {F, G}_{(x,y)}
    pb2_lhs = canonical_bracket(g, h_f .* F_f, h_f .* G_f)
    pb2_rhs = (h_f .^ 2) .* canonical_bracket(g, F_f, G_f)
    err_2d = maximum(abs, pb2_lhs .- pb2_rhs) / maximum(abs, pb2_rhs)
    check("Pointwise flattening identity [hF, hG] = h^2[F, G]: holds in 1D, fails in 2D",
        err_1d < 1e-11 && err_2d > 0.1,
        @sprintf("1D err = %.2e, 2D rel_err = %.3e", err_1d, err_2d))
end

# Dirac reduction non-curative check on Galerkin Vlasov
let s = TensorSplineSpace((3, 3), 2, (Periodic(), Periodic()))
    C = vlasov_galerkin_tensor(s)
    N = nbasis(s)
    z = randn(MersenneTwister(19), N)
    cand_con = collect(6:N)
    z[cand_con] .= 0.0
    P = lie_poisson_matrix(C, z)
    con = maximal_second_class(P, cand_con)
    keep = setdiff(1:N, con)
    Phat, dPhat = reduced_bracket(C, z, keep, con)
    J_proj = project_jacobiator(jacobiator(P, C), dirac_projector(P, keep, con))
    J_red = jacobiator(Phat, dPhat)
    transport_err = maximum(abs, J_red .- J_proj)
    red_jac = jacobi_residual(Phat, dPhat)
    check("Dirac reduction of Galerkin Vlasov: transports Jacobiator and stays O(1)",
        transport_err < 1e-11 && red_jac > 0.05,
        @sprintf("transport_err = %.2e, reduced jac_res = %.3e", transport_err, red_jac))
end

# =============================================================================
# 2. Time-reversal parity splitting and symplectic autoencoder brackets
# =============================================================================
header("2. Time-reversal parity splitting and symplectic autoencoder brackets")

for (label, bc_v) in (("Free()", Free()), ("Dirichlet()", Dirichlet())),
    orth in (false, true)

    s = TensorSplineSpace((4, 5), 2, (Periodic(), bc_v))
    U_plus, U_minus = velocity_parity_projectors(s; orthonormal = orth)
    M = orth ? Matrix(mass_matrix(s)) : Matrix{Float64}(I, nbasis(s), nbasis(s))
    orth_err = max(norm(U_plus' * M * U_plus - I),
        norm(U_minus' * M * U_minus - I),
        norm(U_plus' * M * U_minus))

    f0_even = xv -> (1.0 + 0.3 * cos(xv[1])) * exp(-0.5 * (xv[2] - π)^2)
    J_pp, J_pm, J_mp, J_mm = parity_vlasov_blocks(s, f0_even; orthonormal = orth)
    iso_err = max(maximum(abs, J_pp), maximum(abs, J_mm))

    # Dense reference: the weighted Poisson matrix of the public constructor, projected onto
    # the parity frames (the orthonormal blocks use the raw form M J M')
    Ms = Matrix(mass_matrix(s))
    reference(f0) = (J = vlasov_weighted_poisson_matrix(s, f0); orth ? Ms * J * Ms' : J)
    F = [U_plus U_minus]
    np = size(U_plus, 2)
    J_proj = F' * reference(f0_even) * F
    block_err = max(norm(J_pm - J_proj[1:np, (np + 1):end]),
        norm(J_mp - J_proj[(np + 1):end, 1:np]))

    f0_skew = xv -> (1.0 + 0.3 * cos(xv[1])) * exp(-0.5 * (xv[2] - π - 0.7)^2)
    J_pp_skew, J_pm_skew, J_mp_skew, J_mm_skew = parity_vlasov_blocks(
        s, f0_skew; orthonormal = orth)
    skew_mag = max(maximum(abs, J_pp_skew), maximum(abs, J_mm_skew))
    J_proj_skew = F' * reference(f0_skew) * F
    skew_err = max(norm(J_pp_skew - J_proj_skew[1:np, 1:np]),
        norm(J_pm_skew - J_proj_skew[1:np, (np + 1):end]),
        norm(J_mp_skew - J_proj_skew[(np + 1):end, 1:np]),
        norm(J_mm_skew - J_proj_skew[(np + 1):end, (np + 1):end]))

    b_par = parity_vlasov_bracket(s, f0_even; orthonormal = orth)
    par_err = norm(Matrix(b_par.P) - J_proj)

    tag = @sprintf("bc=%s, orth=%s", label, orth)
    check("Parity projectors ($tag): orthonormal and isotropic on v-even f0",
        orth_err < 1e-13 && iso_err < 1e-13 && block_err < 1e-12 && par_err < 1e-12,
        @sprintf("orth = %.2e, max(|J++|,|J--|) = %.2e, block = %.2e, bracket = %.2e",
            orth_err, iso_err, block_err, par_err))
    check("Parity blocks ($tag): v-skew f0 breaks isotropy (control)",
        skew_mag > 1e-2 && skew_err < 1e-12,
        @sprintf("max(|J++_skew|,|J--_skew|) = %.3e, block = %.2e", skew_mag, skew_err))
end

# O(r) × O(r) DLRA parity bracket and nonlinear symplectic shear + PSDLayer pullback
let s = TensorSplineSpace((4, 6), 2, (Periodic(), Free()))
    f0_even = xv -> (1.0 + 0.2 * sin(xv[1])) * exp(-0.5 * (xv[2] - π)^2)
    _, A_full, _, _ = parity_vlasov_blocks(s, f0_even; orthonormal = true)
    nx, nv = size(s)
    m_minus = div(nv, 2)
    m_plus = nv - m_minus
    r = 2
    Ux = Matrix(qr(randn(MersenneTwister(1), nx, r)).Q)
    Uv_p = Matrix(qr(randn(MersenneTwister(2), m_plus, r)).Q)
    Uv_m = Matrix(qr(randn(MersenneTwister(3), m_minus, r)).Q)
    b_dlra = dlra_parity_vlasov_bracket(A_full, Ux, Uv_p, Uv_m)
    # The DLRA bracket is the parity bracket pulled back along the Stiefel embedding
    P_plus = kron(Uv_p, Ux)
    P_minus = kron(Uv_m, Ux)
    E = [P_plus zeros(size(P_plus, 1), r^2); zeros(size(P_minus, 1), r^2) P_minus]
    dlra_err = norm(Matrix(b_dlra.P) - E' * Matrix(parity_vlasov_bracket(A_full).P) * E)
    check("O(r)×O(r) DLRA parity Vlasov bracket: pullback of the parity bracket",
        size(b_dlra.P) == (2 * r^2, 2 * r^2) && dlra_err < 1e-13,
        @sprintf("size = %d×%d, ||P_dlra - Eᵀ P_par E|| = %.2e",
            size(b_dlra.P, 1), size(b_dlra.P, 2), dlra_err))

    # Nonlinear Darboux dressing: compose a symplectic Stiefel embedding A ∈ Sp(2n, 2N)
    # with an upper/lower symplectic gradient shear and an affine reference shift z_ref
    N_lat = 4
    n_lat = 2
    J2N = canonical_vlasov_bracket(N_lat).P
    J2n = canonical_vlasov_bracket(n_lat).P
    U_orth = Matrix(qr(randn(MersenneTwister(5), N_lat, n_lat)).Q)
    A_sp = [U_orth zeros(N_lat, n_lat); zeros(N_lat, n_lat) U_orth]
    K_sym = Symmetric(randn(MersenneTwister(6), N_lat, N_lat))
    # Symplectic shear ψ(Q, P) = (Q, P + K_sym * tanh.(K_sym * Q)), the P-shift being the gradient
    # of Σ log cosh((K_sym Q)_i), has Jacobian [I 0; D_shear I] ∈ Sp(2N) with the symmetric
    # D_shear = K_sym * Diagonal(sech.(K_sym * Q) .^ 2) * K_sym
    decoder(ξ) = (z = A_sp * ξ; Q = z[1:N_lat];
        vcat(Q, z[(N_lat + 1):end] .+ K_sym * tanh.(K_sym * Q)))
    ξ = randn(MersenneTwister(7), 2 * n_lat)
    Q_lin = (A_sp * ξ)[1:N_lat]
    D_shear = K_sym * Diagonal(sech.(K_sym * Q_lin) .^ 2) * K_sym
    D_psi = [Matrix{Float64}(I, N_lat, N_lat) zeros(N_lat, N_lat);
             D_shear Matrix{Float64}(I, N_lat, N_lat)]
    D_dec = D_psi * A_sp
    # the analytic Jacobian against central differences of the decoder itself
    h = 1e-6
    D_fd = reduce(hcat,
        [(decoder(ξ + h * e) - decoder(ξ - h * e)) / 2h
         for e in eachcol(Matrix{Float64}(I, 2 * n_lat, 2 * n_lat))])
    jac_err = norm(D_dec - D_fd)
    symp_err = norm(D_dec' * J2N * D_dec - J2n)
    check("Nonlinear symplectic shear + PSDLayer decoder: exact canonical pullback J_{2n}",
        jac_err < 1e-8 && symp_err < 1e-14,
        @sprintf("||DΨ - DΨ_fd|| = %.2e, ||DΨ^T J_{2N} DΨ - J_{2n}|| = %.2e",
            jac_err, symp_err))
end

# =============================================================================
# 3. Canonical Clebsch potentials and the Discrete Clebsch Non-Submersion Theorem
# =============================================================================
header("3. Canonical Clebsch potentials and Discrete Clebsch Non-Submersion Theorem")

let s = TensorSplineSpace((3, 3), 2, (Periodic(), Periodic()))
    N = nbasis(s)
    B = clebsch_vlasov_tensor(s)
    b_can = canonical_vlasov_bracket(N)
    rng = MersenneTwister(11)
    Q = randn(rng, N)
    P = randn(rng, N)
    z = vcat(Q, P)
    c0 = ones(N)
    c = clebsch_vlasov_map(B, Q, P, c0)
    b_int = basis_integrals(s)
    mass_err = abs(dot(b_int, c) - dot(b_int, c0))
    J_can = [zeros(N, N) Matrix{Float64}(I, N, N); -Matrix{Float64}(I, N, N) zeros(N, N)]
    can_err = norm(Matrix(b_can.P) - J_can)

    check(
        "Canonical Clebsch phase space (R^{2N}, J_{2N}): canonical matrix and mass conservation",
        can_err == 0 && mass_err < 1e-13,
        @sprintf("||J_{2N} - [0 I; -I 0]|| = %.1e, |∫f_h - ∫f_{0,h}| = %.2e",
            can_err, mass_err))

    Π = clebsch_pushforward_matrix(B, Q, P)
    Π_comm = [-dot(Q, (B[m, :, :] * B[n, :, :] - B[n, :, :] * B[m, :, :]) * P)
              for m in 1:N, n in 1:N]
    comm_err = norm(Π - Π_comm)
    check("Clebsch pushforward identity: Π_{mn}(Q, P) == -Q^T [B_m, B_n] P",
        comm_err < 1e-13, @sprintf("||Π - (-Q^T[B_m,B_n]P)|| = %.2e", comm_err))

    # Non-submersion on spline space: compare against J_LP(c - c0) and move along ker Dc(Q, P)
    C_gal = vlasov_galerkin_tensor(s)
    J_lp = lie_poisson_matrix(C_gal, c - c0)
    lp_rel_diff = min(norm(Π - J_lp), norm(Π + J_lp)) / norm(Π)

    # Jacobian Dc(Q, P) is N × 2N: since 2N > N, ker Dc(Q, P) has dimension >= N.
    # For a small step along ker Dc, solve the quadratic constraint to stay on the exact fibre
    # c(Q', P') == c(Q, P) to machine precision, and measure how much Π(Q', P') changes.
    Dc = Matrix{Float64}(undef, N, 2N)
    for m in 1:N
        Bm = @view B[m, :, :]
        Dc[m, 1:N] = Bm * P
        Dc[m, (N + 1):(2N)] = Bm' * Q
    end
    null_vecs = nullspace(Dc)
    δz = 0.25 .* null_vecs[:, 1]
    z_fib = copy(z) + δz
    for _ in 1:6
        r_c = clebsch_vlasov_map(B, z_fib[1:N], z_fib[(N + 1):(2N)], c0) - c
        Dc_f = Matrix{Float64}(undef, N, 2N)
        for m in 1:N
            Bm = @view B[m, :, :]
            Dc_f[m, 1:N] = Bm * z_fib[(N + 1):(2N)]
            Dc_f[m, (N + 1):(2N)] = Bm' * z_fib[1:N]
        end
        z_fib .-= Dc_f \ r_c
    end
    c_fib = clebsch_vlasov_map(B, z_fib[1:N], z_fib[(N + 1):(2N)], c0)
    fibre_c_err = norm(c_fib - c)
    Π_fib = clebsch_pushforward_matrix(B, z_fib[1:N], z_fib[(N + 1):(2N)])
    fibre_Π_diff = norm(Π_fib - Π) / norm(Π)

    check("Discrete Clebsch Non-Submersion Theorem on splines: Π varies along fibres of c",
        lp_rel_diff > 0.1 && fibre_c_err < 1e-12 && fibre_Π_diff > 1e-2,
        @sprintf("min||Π±J_LP||/||Π|| = %.3f, ||Δc|| = %.2e, ||ΔΠ||/||Π|| = %.3e",
            lp_rel_diff, fibre_c_err, fibre_Π_diff))

    # Positive control: when B is a closed Lie algebra (so(3)), Π depends ONLY on c(Q, P)
    C_so3 = Float64.(so3())
    q3 = [0.6, -0.4, 0.9]
    p3 = [-0.3, 0.8, 0.5]
    c3 = clebsch_vlasov_map(C_so3, q3, p3)
    Π_so3 = clebsch_pushforward_matrix(C_so3, q3, p3)
    so3_err = norm(Π_so3 - lie_poisson_matrix(C_so3, c3))
    check("Clebsch submersion positive control (closed so(3)): Π(Q, P) == J_LP(c(Q, P))",
        so3_err < 1e-14, @sprintf("||Π - J_LP(c)|| = %.2e", so3_err))
end

# =============================================================================
# 4. Matrix quantisation on u(N)*, Weyl symbols, and spectral pushforwards
# =============================================================================
header("4. Matrix quantisation on u(N)*, Weyl symbols, and spectral pushforwards")

for N in (3, 4)
    C_un = un_structure_constants(N; ħ = 0.5)
    sc_un = structure_constant_residual(C_un)
    w = randn(MersenneTwister(20 + N), N^2)
    J_un = lie_poisson_matrix(C_un, w)
    jac_un = jacobi_residual(J_un, C_un)
    check("u($N)* von Neumann-Poisson bracket: exact Lie-Poisson structure",
        is_antisymmetric_c(C_un) && sc_un < 1e-14 && jac_un < 1e-14,
        @sprintf("sc_res = %.2e, jac_res = %.2e", sc_un, jac_un))
end

# Weyl quantisation: exact inner block for degree <= 1, Wigner O(ħ^2) dispersion at degree 2
let K = 10, ħ = 0.2
    # h(x, v) = v^2 / 2 + cos(2x), f(x, v) = cos(x) v^3
    # {f, h}_{(x, v)} = ∂_x f ∂_v h - ∂_v f ∂_x h = -sin(x) v^4 + 6 cos(x) sin(2x) v^2
    # Wigner third-order Moyal term: -(ħ^2 / 24) ∂_x^3 h ∂_v^3 f = -(ħ^2 / 24) (8 sin(2x)) (6 cos(x))
    #   = -2 ħ^2 sin(2x) cos(x) = -ħ^2 (sin(3x) + sin(x))
    h_modes = Dict(
        (0, 2) => 0.5 + 0.0im,
        (2, 0) => 0.5 + 0.0im,
        (-2, 0) => 0.5 + 0.0im
    )
    f_modes = Dict(
        (1, 3) => 0.5 + 0.0im,
        (-1, 3) => 0.5 + 0.0im
    )
    # Continuum Poisson bracket {f, h}:
    # -sin(x) v^4 = (i/2)(e^{ix} - e^{-ix}) v^4
    # 6 cos(x) sin(2x) v^2 = 3 (sin(3x) + sin(x)) v^2 = (-3i/2)(e^{3ix} - e^{-3ix} + e^{ix} - e^{-ix}) v^2
    pb_modes = Dict(
        (1, 4) => 0.5im,
        (-1, 4) => -0.5im,
        (3, 2) => -1.5im,
        (-3, 2) => 1.5im,
        (1, 2) => -1.5im,
        (-1, 2) => 1.5im
    )
    wigner_modes = Dict(
        (3, 0) => -0.5im * ħ^2,
        (-3, 0) => 0.5im * ħ^2,
        (1, 0) => -0.5im * ħ^2,
        (-1, 0) => 0.5im * ħ^2
    )
    Th = weyl_vlasov_matrix(h_modes, K, ħ)
    Tf = weyl_vlasov_matrix(f_modes, K, ħ)
    Tpb = weyl_vlasov_matrix(pb_modes, K, ħ)
    Twig = weyl_vlasov_matrix(wigner_modes, K, ħ)
    comm_fh = (Tf * Th - Th * Tf) / (im * ħ)
    inner = (K + 1 - 5):(K + 1 + 5)
    pb_only_err = maximum(abs, comm_fh[inner, inner] - Tpb[inner, inner])
    full_moyal_err = maximum(abs, comm_fh[inner, inner] -
                                  (Tpb[inner, inner] + Twig[inner, inner]))
    check(
        "Weyl quantisation on S^1×R: exact Moyal-Wigner O(ħ^2) dispersion on inner block",
        pb_only_err > 1e-3 && full_moyal_err < 1e-13,
        @sprintf("||comm - T({f,h})|| = %.3e, ||comm - T({f,h} + Wigner)|| = %.2e",
            pb_only_err, full_moyal_err))
end

# Logarithmic-mean (BKM) spectral pushforward vs naive matrix structure-function bracket on u(3)*
let N = 3, M = 9
    rng = MersenneTwister(77)
    w = 0.25 .* randn(rng, M)
    J_bkm, dJ_bkm = spectral_pushforward_bracket(
        λ -> 0.5 * exp(2λ), λ -> exp(2λ), λ -> 2 * exp(2λ), w)
    jac_bkm = jacobi_residual(J_bkm, dJ_bkm)

    w_big = [BigFloat(2 * (-1)^k) / BigFloat(10 * (k + 1)) for k in 1:M]
    J_bkm_big, dJ_bkm_big = spectral_pushforward_bracket(
        λ -> exp(2λ) / 2, λ -> exp(2λ), λ -> 2 * exp(2λ), w_big)
    jac_bkm_big = Float64(jacobi_residual(J_bkm_big, dJ_bkm_big))

    # Naive matrix structure-function bracket tr(W^2 [F_W, G_W]) on u(3)* (analytical derivative)
    E = un_hermitian_basis(N)
    C_un = un_structure_constants(N)
    W = sum(w[a] .* E[a] for a in 1:M)
    W2 = W * W
    c_quad = [real(tr(E[a] * W2)) for a in 1:M]
    J_naive = lie_poisson_matrix(C_un, c_quad)
    dJ_naive = zeros(Float64, M, M, M)
    for k in 1:M
        dW2_k = E[k] * W + W * E[k]
        dc_k = [real(tr(E[a] * dW2_k)) for a in 1:M]
        dJ_naive[k, :, :] .= lie_poisson_matrix(C_un, dc_k)
    end
    jac_naive = jacobi_residual(J_naive, dJ_naive)

    check(
        "Logarithmic-mean (BKM) spectral pushforward on u(3)*: exact Jacobi (Float64 & BigFloat)",
        jac_bkm < 1e-14 && jac_bkm_big < 1e-70,
        @sprintf("jac_f64 = %.2e, jac_big = %.2e", jac_bkm, jac_bkm_big))
    check(
        "Naive matrix structure-function tr(W^2[F_W,G_W]) on u(3)*: fails Jacobi (control)",
        jac_naive > 0.1, @sprintf("jac_res = %.3e", jac_naive))

    # Dirac reduction of exact u(3)* Lie-Poisson bracket preserves Jacobi
    w_dir = copy(w)
    cand_con = [6, 7, 8, 9]
    w_dir[cand_con] .= 0.0
    P_un = lie_poisson_matrix(C_un, w_dir)
    con_un = maximal_second_class(P_un, cand_con)
    keep_un = setdiff(1:M, con_un)
    Phat_un, dPhat_un = reduced_bracket(C_un, w_dir, keep_un, con_un)
    jac_red_un = jacobi_residual(Phat_un, dPhat_un)
    check("Dirac reduction of exact u(3)* bracket: preserves Jacobi to roundoff",
        jac_red_un < 1e-13, @sprintf("nk = %d, jac_res = %.2e", length(keep_un),
            jac_red_un))
end

# =============================================================================
# 5. Semidirect-product Vlasov moment / waterbag brackets (exact Rational{BigInt})
# =============================================================================
header("5. Semidirect-product Vlasov moment brackets in exact Rational{BigInt}")

let
    Ka_per = Rational{BigInt}[0 1 0 -1; -1 0 1 0; 0 -1 0 1; 1 0 -1 0] .// 2
    Ka_wall = Rational{BigInt}[-3 4 -1 0; -2 0 2 0; 0 -2 0 2; 0 1 -4 3] .// 5
    Kb_mag = Rational{BigInt}[0 2 -1 0; -2 0 0 1; 1 0 0 -3; 0 -1 3 0] .// 7
    ρ_q = Rational{BigInt}[2, 3, 5, 4] .// 2
    m_q = Rational{BigInt}[1, -3, 2, 5] .// 3

    J_per, dJ_per = vlasov_moment_flat_bracket(Ka_per, ρ_q, m_q)
    J_wall, dJ_wall = vlasov_moment_flat_bracket(Ka_wall, ρ_q, m_q; Kb = Kb_mag)
    J_dir, dJ_dir = vlasov_moment_direct_bracket(Ka_per, ρ_q, m_q)

    jac_per = jacobi_residual(J_per, dJ_per)
    jac_wall = jacobi_residual(J_wall, dJ_wall)
    jac_dir = jacobi_residual(J_dir, dJ_dir)

    check("Pointwise-flat Vlasov moment bracket (periodic Ka): exact Jacobi == 0",
        J_per == -transpose(J_per) && iszero(jac_per),
        "jac_res = $(jac_per)")
    check(
        "Pointwise-flat Vlasov moment bracket (wall-bounded Ka + magnetic Kb): exact Jacobi == 0",
        J_wall == -transpose(J_wall) && iszero(jac_wall),
        "jac_res = $(jac_wall)")
    check("Direct-collocation Vlasov moment bracket: violates Jacobi at O(1) (control)",
        !iszero(jac_dir) && Float64(jac_dir) > 0.05,
        @sprintf("jac_res = %.4f (%s)", Float64(jac_dir), string(jac_dir)))
end

summary("verify_vlasov_brackets.jl")
