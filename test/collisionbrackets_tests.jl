using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, Dirichlet, Free, Periodic, (..)
using Test

# `Q_2` from its definition, `|z|² I - z ⊗ z`. Every brute-force reference below builds the
# kernel from this rather than from the perp identity `Q_2(z) = z^⊥ ⊗ z^⊥`, so the checks
# test that identity and the moment expansion together instead of one rearrangement of the
# collapsed form against another.
Q2def(z) = (z[1]^2 + z[2]^2) * I(2) - z * z'

# The state the collapsed assembly reads, recomputed here from the exported interface so
# that the references do not go through the code under test.
function bracket_samples(b, û)
    s = b.space
    x = quadrature_nodes(s)
    u = field(s, û, (0, 0))
    M = [b.mobility(x[r], u[r]) for r in eachindex(u)]
    φ̂ = GeometricBrackets._generator(b, û)
    ∇φ = (field(s, φ̂, (1, 0)), field(s, φ̂, (0, 1)))
    (; M, c = M .* quadrature_weights(s) .* b.density, ∇φ)
end

"""
The `O(N_q²)` double sum for the weak-form operator, straight from the definition

    A_KL = ½ ∬ κ (∇Φ_K(x) - ∇Φ_K(x'))ᵀ Q(∇φ(x) - ∇φ(x')) (∇Φ_L(x) - ∇Φ_L(x')) dμ' dμ .

`kernel` is swapped for the degeneracy control and `inner` for the measure control: it
returns the inner kernel weight and may depend on the outer node, which is exactly what an
`x`-dependent measure does and what the collapse forbids.
"""
function double_sum_operator(b, û; kernel = Q2def, inner = (r, r′, c) -> c[r′])
    s = b.space
    smp = bracket_samples(b, û)
    c, ∇φ = smp.c, smp.∇φ
    P = (Matrix(basis_values(s, (1, 0))), Matrix(basis_values(s, (0, 1))))
    N, Nq = nbasis(s), length(c)
    A = zeros(N, N)
    for r in 1:Nq, r′ in 1:Nq

        K = kernel([∇φ[1][r] - ∇φ[1][r′], ∇φ[2][r] - ∇φ[2][r′]])
        D = (P[1][:, r] .- P[1][:, r′], P[2][:, r] .- P[2][:, r′])
        f = 0.5 * c[r] * inner(r, r′, c)
        for l in 1:2, k in 1:2

            A .+= (f * K[k, l]) .* (D[k] * D[l]')
        end
    end
    return A
end

"`𝔻_s` at every quadrature node by brute force, the differences taken before anything is summed."
function double_sum_diffusion(b, û)
    smp = bracket_samples(b, û)
    Nq = length(smp.c)
    𝔻 = [zeros(Nq) for _ in 1:2, _ in 1:2]
    for r in 1:Nq
        D = zeros(2, 2)
        for r′ in 1:Nq
            D .+= smp.c[r′] .*
                  Q2def([smp.∇φ[1][r] - smp.∇φ[1][r′], smp.∇φ[2][r] - smp.∇φ[2][r′]])
        end
        for l in 1:2, k in 1:2

            𝔻[k, l][r] = D[k, l]
        end
    end
    return 𝔻
end

# `Σ` formed as `M₂ - m₀ β̄ ⊗ β̄` rather than accumulated centred: the one variant of the
# collapse that is algebraically identical and numerically wrong.
function uncentred_diffusion(b, û)
    st = GeometricBrackets._collision_state(b, û)
    smp = bracket_samples(b, û)
    β = (-smp.∇φ[2], smp.∇φ[1])
    c, m₀, γ = st.c, st.m₀, st.γ
    β̄ = ntuple(k -> dot(c, β[k]) / m₀, 2)
    Σ = [dot(c, β[k] .* β[l]) - m₀ * β̄[k] * β̄[l] for k in 1:2, l in 1:2]
    [@.(m₀ * γ[k] * γ[l] + Σ[k, l]) for k in 1:2, l in 1:2]
end

pointwise(𝔻, r) = Symmetric([𝔻[1, 1][r] 𝔻[1, 2][r]; 𝔻[2, 1][r] 𝔻[2, 2][r]])
notpsd(𝔻, Nq) = count(r -> (λ = eigvals(pointwise(𝔻, r)); λ[1] < -1e-12 * λ[2]), 1:Nq)
function tensor_error(a, b)
    maximum(maximum(abs, a[k, l] .- b[k, l]) for k in 1:2, l in 1:2) /
    maximum(maximum(abs, b[k, l]) for k in 1:2, l in 1:2)
end

# The central-difference Jacobian of `metric_matrix`, as the tensor `dG[m,i,j]`.
function difference_derivative(b, û, ε)
    N = length(û)
    dG = zeros(eltype(û), N, N, N)
    for m in 1:N
        e = zeros(N)
        e[m] = ε
        dG[m, :, :] = (metric_matrix(b, û + e) - metric_matrix(b, û - e)) / (2ε)
    end
    return dG
end

# The two spaces the section-5 experiments live on, with the two mobilities that exercise
# both kinds of dependence: `M = 1` on the periodic torus, and a state-dependent `M(x, u)`
# on the homogeneous-Dirichlet square under the non-Lebesgue measure `dμ = dx / (1 + x₁)`.
function periodic_bracket()
    s = TensorSplineSpace((4, 4), 2)
    M = Matrix(mass_matrix(s))
    s, CollisionBracket(s, (Matrix(stiffness_matrix(s)) + 0.5M) \ M)
end

function dirichlet_bracket()
    s = TensorSplineSpace(ntuple(_ -> UniformMesh(3, 1.0), 2), 2, Dirichlet())
    Λ = Matrix(stiffness_matrix(s)) \ Matrix(mass_matrix(s))
    s,
    CollisionBracket(s, Λ; mobility = (x, u) -> 0.5 + u^2,
        mobility_derivative = (x, u) -> 2u, density = x -> 1 / (1 + x[1]))
end

@testset "$(rpad("Collision Bracket Tests",80))" begin
    rng = MersenneTwister(0x7ac91d02)

    @testset "$(rpad("the collapsed operator IS the O(N_q²) double sum, to round-off",76))" begin
        s, b = periodic_bracket()
        û = randn(rng, nbasis(s))
        A = metric_operator(b, û)
        Ad = double_sum_operator(b, û)

        @test b isa MetricBracket{Float64}
        @test size(b) == (nbasis(s), nbasis(s))
        @test maximum(abs, Ad) > 1.0                    # the reference is not trivially zero
        @test maximum(abs, A - Ad) < 1e-12 * maximum(abs, Ad)

        # the same, with a state-dependent mobility and a non-Lebesgue measure — the two
        # cases the collapse is claimed to survive, and the ones an implementation that
        # folded dμ in the wrong place would fail
        s2, b2 = dirichlet_bracket()
        û2 = randn(rng, nbasis(s2))
        A2 = metric_operator(b2, û2)
        A2d = double_sum_operator(b2, û2)
        @test maximum(abs, A2d) > 1e-5
        @test maximum(abs, A2 - A2d) < 1e-12 * maximum(abs, A2d)

        # the measure is genuinely non-constant, so the check above is not the Lebesgue one
        # under another name
        @test maximum(b2.density) / minimum(b2.density) > 1.5
    end

    @testset "$(rpad("the bracket is SYMMETRIC and positive SEMI-definite, never definite",76))" begin
        s, b = dirichlet_bracket()
        N = nbasis(s)
        û = randn(rng, N)

        G = metric_matrix(b, û)
        λ = eigvals(Symmetric(G))
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test issymmetric(b, û)
        @test minimum(λ) > -1e-12 * maximum(λ)
        @test ispositive_semidefinite(b, û)
        # singular, not definite: the Hamiltonian is in the kernel by construction
        @test minimum(λ) < 1e-12 * maximum(λ)
        # and it really depends on the state, so none of the above is a statement about a
        # constant matrix
        @test maximum(abs, metric_matrix(b, randn(rng, N)) - G) > 1e-3 * maximum(abs, G)

        # the control that must fail: `Q₂` with the wrong sign is still symmetric and is
        # negative semi-definite, which is the sign of the entropy production flipped
        Abad = -double_sum_operator(b, û)
        @test maximum(abs, Abad - Abad') < 1e-12 * maximum(abs, Abad)
        @test maximum(eigvals(Symmetric(Abad))) < 1e-12 * maximum(abs, Abad)
    end

    @testset "$(rpad("the Hamiltonian is in the KERNEL at the quadrature level",76))" begin
        s, b = dirichlet_bracket()
        N = nbasis(s)
        û = randn(rng, N)
        φ̂ = GeometricBrackets._generator(b, û)

        # Q₂(z) z = 0 holds at every *pair* of quadrature points, so this is exact at the
        # discrete level and not only in the continuum
        @test degeneracy_residual(b, û) < 1e-13
        @test degeneracy_residual(b, û, randn(rng, N)) > 0.1

        # the control that must fail, and the reason `degeneracy_residual` exists: drop the
        # ⊥ and use z ⊗ z instead of z^⊥ ⊗ z^⊥. It stays symmetric and positive
        # semi-definite — a Gram matrix is a Gram matrix — and only the degeneracy notices.
        Anp = double_sum_operator(b, û; kernel = z -> z * z')
        @test maximum(abs, Anp - Anp') < 1e-12 * maximum(abs, Anp)
        @test minimum(eigvals(Symmetric(Anp))) > -1e-12 * maximum(abs, Anp)
        @test maximum(abs, Anp * φ̂) > 0.1 * maximum(abs, Anp) * maximum(abs, φ̂)

        # the second control: the local half alone. `𝔻_s ∇φ(x) ≠ 0` — only the *difference*
        # of gradients is annihilated — so it is the nonlocal cross term that carries the
        # degeneracy, and an implementation that dropped it would still be symmetric and
        # positive semi-definite.
        st = GeometricBrackets._collision_state(b, û)
        𝔻 = GeometricBrackets._diffusion_tensor(st)
        ϱ = b.density .* st.M
        Aloc = tensor_weighted_matrix(s, [ϱ .* 𝔻[k, l] for k in 1:2, l in 1:2])
        @test minimum(eigvals(Symmetric(Matrix(Aloc)))) > -1e-12 * maximum(abs, Aloc)
        @test maximum(abs, Aloc * φ̂) > 0.1 * maximum(abs, Aloc) * maximum(abs, φ̂)
    end

    @testset "$(rpad("metric_apply agrees with the assembled matrix through 𝔽_s",76))" begin
        for (s, b) in (periodic_bracket(), dirichlet_bracket())
            N = nbasis(s)
            û = randn(rng, N)
            c = randn(rng, N)
            G = metric_matrix(b, û)
            # two independent routes to the same operator: the fourteen scalar moments,
            # against the 𝕂-indexed factorisation of the assembled cross term
            @test maximum(abs, metric_apply(b, û, c) - G * c) < 1e-12 * maximum(abs, G * c)
            @test metric_operator(b, û) ≈ mass_matrix(s) * G * mass_matrix(s)
        end
    end

    @testset "$(rpad("RECENTRING is a correctness requirement, not a refinement",76))" begin
        # φ = 3x₁ + ε g(x): a large mean gradient with a small variation, which is what a
        # relaxed Grad-Shafranov state looks like. A clamped basis, because a linear
        # function lies in neither a periodic nor a homogeneous-Dirichlet space.
        s = TensorSplineSpace(ntuple(_ -> UniformMesh(4, 1.0), 2), 3, Free())
        N, Nq = nbasis(s), length(quadrature_weights(s))
        û = zeros(N)

        # The claim is about CONDITIONING, so it is read off a sweep in the spread ε and not
        # off one value of it. The amplification the centring removes is |α|² m₀ / ‖𝔻ₛ‖, which
        # grows as ε falls; a single ε would only establish that the two forms differ there,
        # and could do so for any reason.
        εs = (1e-5, 1e-7, 1e-9, 1e-11)
        bs = [CollisionBracket(s, project(s, p -> 3p[1] +
                                                  ε * (sin(4p[1]) * cos(3p[2]) + p[2]^2)))
              for ε in εs]
        refs = [double_sum_diffusion(b, û) for b in bs]
        centred = [GeometricBrackets._diffusion_tensor(b, û) for b in bs]
        raw = [uncentred_diffusion(b, û) for b in bs]
        ec = [tensor_error(centred[i], refs[i]) for i in eachindex(εs)]
        er = [tensor_error(raw[i], refs[i]) for i in eachindex(εs)]

        # centred: round-off at every spread, and FLAT across six decades of ε -- the
        # conditioning never reaches it
        @test maximum(ec) < 1e-12
        @test maximum(ec) / minimum(ec) < 10
        @test all(i -> notpsd(centred[i], Nq) == 0, eachindex(εs))

        # uncentred: `M₂ - m₀ β̄ ⊗ β̄` is the same algebra and reinstates exactly the
        # cancellation the centring removes, so it degrades monotonically until nothing is
        # left. Already wrong by three orders more than the centred form at the LOOSEST
        # spread here, and by the tightest it has no correct digits at all.
        @test issorted(er)
        @test er[1] > 1e3 * ec[1]
        @test er[end] / er[1] > 1e6
        @test er[end] > 1e3

        # What is deliberately NOT asserted here: that the uncentred 𝔻ₛ goes *indefinite*. It
        # does, and that is the failure that flips the sign of the entropy production -- but
        # on any one φ whether it lands below zero or just above is decided by summation
        # order, and the decision changes with bounds checking: a majority of seeded draws
        # lose semi-definiteness under `--check-bounds=auto` and none at all under
        # `--check-bounds=yes`, which is what `Pkg.test()` runs. It is a statement about a
        # distribution, and it is measured where one can be: `verify_metric_collapse.jl`
        # sweeps it over draws, 45/200 at spread 1e-7 rising to 180/200 at 1e-10, against
        # 0/1800 for the centred form.

        # and the loss is a property of the conditioning, not of the algebra: at a spread
        # where the amplification is mild both forms agree
        b0 = CollisionBracket(s, project(s, p -> 3p[1] + sin(4p[1]) * cos(3p[2]) + p[2]^2))
        @test tensor_error(uncentred_diffusion(b0, û), double_sum_diffusion(b0, û)) < 1e-10
    end

    @testset "$(rpad("an x-DEPENDENT measure breaks the collapse",76))" begin
        s, b = dirichlet_bracket()
        û = randn(rng, nbasis(s))
        x = quadrature_nodes(s)
        A = metric_operator(b, û)

        # dμ = dx / (1 + x₁) is x-independent and harmless: it is absorbed into the
        # quadrature weights and never sees the outer point
        @test maximum(abs, A - double_sum_operator(b, û)) < 1e-12 * maximum(abs, A)
        # give the same weight a dependence on the outer point and one fixed set of moments
        # cannot serve every x, which is what the hypothesis actually forbids
        Axd = double_sum_operator(b, û; inner = (r, r′, c) -> c[r′] / (1 + x[r][1]))
        @test maximum(abs, A - Axd) > 0.1 * maximum(abs, Axd)
    end

    @testset "$(rpad("metric_derivative is ANALYTIC in both dependences",76))" begin
        s, b = dirichlet_bracket()
        N = nbasis(s)
        û = randn(rng, N)

        # both dependences at once: 𝔾 is quadratic in φ̂ = Λû and bilinear in M(x, u_h(x))
        dG = metric_derivative(b, û)
        fd = difference_derivative(b, û, 1e-5)
        # The floor is set by the difference scheme, not by the derivative: entries of order
        # 1 differenced at ε = 10⁻⁵ carry a relative error around 10⁻¹¹, and any tighter
        # tolerance would be measuring the step size rather than the formula.
        @test maximum(abs, dG - fd) < 1e-8 * maximum(abs, dG)
        # the control: the tolerance discriminates. A derivative wrong by one per cent —
        # far smaller than any plausible index or sign error — fails it.
        @test maximum(abs, 1.01 * dG - fd) > 1e-8 * maximum(abs, dG)

        # the mobility half on its own: a prescribed φ, so the only state dependence left
        # is M(x, u), and it must still be found
        bm = CollisionBracket(s, GeometricBrackets._generator(b, û);
            mobility = (x, u) -> 0.5 + u^2, mobility_derivative = (x, u) -> 2u,
            density = b.density)
        dGm = metric_derivative(bm, û)
        @test maximum(abs, dGm) > 1e-3
        @test maximum(abs, dGm - difference_derivative(bm, û, 1e-5)) <
              1e-8 * maximum(abs, dGm)

        # and with neither dependence the bracket is constant, which is asserted rather
        # than assembled N times
        bc = CollisionBracket(s, GeometricBrackets._generator(b, û); density = b.density)
        @test metric_derivative(bc, û) == zeros(N, N, N)
        @test metric_matrix(bc, û) ≈ metric_matrix(bc, randn(rng, N))
    end

    @testset "$(rpad("the Grad-Shafranov residual is exactly CUBIC in the dofs",76))" begin
        # §5.5: s(r,y) = y²/2(Cr²+D) with dμ = dr dz / r, so M = Cr² + D is independent of
        # u and ∂S/∂û is linear. Both follow from eq:M-condition, M ∂²_y s = 1, and both
        # are checked here rather than asserted.
        C, D = 0.6, 0.2
        s = TensorSplineSpace(
            (UniformMesh(3, (1.0, 3.0)), UniformMesh(3, (-2.0, 2.0))), 2, Dirichlet())
        N = nbasis(s)
        x = quadrature_nodes(s)
        ρ = [1 / p[1] for p in x]
        M = [C * p[1]^2 + D for p in x]
        Λ = Matrix(stiffness_matrix(s)) \ Matrix(mass_matrix(s))
        b = CollisionBracket(s, Λ; mobility = (p, u) -> C * p[1]^2 + D,
            mobility_derivative = 0, density = p -> 1 / p[1])

        # ∂S/∂û = ∫ Φ_K ∂_y s dμ with ∂_y s = u / (Cr² + D)
        entropy_gradient(û) = basis_values(s, (0, 0)) *
                              (quadrature_weights(s) .* ρ .*
                               field(s, û, (0, 0)) ./ M)
        residual(û) = metric_apply(b, û, entropy_gradient(û))

        û₀ = randn(rng, N)
        d = randn(rng, N)
        g = entropy_gradient(û₀)
        # linear in û, which is the half of the claim that eq:M-condition supplies
        @test maximum(abs, entropy_gradient(û₀ + 2d) - 2entropy_gradient(û₀ + d) + g) <
              1e-12 * maximum(abs, g)

        f = [residual(û₀ + t * d) for t in 0:5]
        Δ(v) = [v[i + 1] - v[i] for i in 1:(length(v) - 1)]
        third = Δ(Δ(Δ(f)))
        fourth = Δ(third)
        scale = maximum(maximum(abs, v) for v in third)
        @test scale > 1e-3                              # the cubic term is not degenerate
        @test maximum(maximum(abs, v) for v in fourth) < 1e-9 * scale
        @test maximum(abs, third[1] - third[2]) < 1e-9 * maximum(abs, third[1])
        @test degeneracy_residual(b, û₀) < 1e-13

        # the control: make M depend on u and the residual is quintic, so the fourth
        # difference no longer vanishes. Cubicity is a consequence of eq:M-condition here,
        # not of the discretisation.
        bq = CollisionBracket(s, Λ; mobility = (p, u) -> C * p[1]^2 + D + u^2,
            mobility_derivative = (p, u) -> 2u, density = p -> 1 / p[1])
        fq = [metric_apply(bq, û₀ + t * d, entropy_gradient(û₀ + t * d)) for t in 0:5]
        thirdq = Δ(Δ(Δ(fq)))
        @test maximum(maximum(abs, v) for v in Δ(thirdq)) >
              0.1 * maximum(maximum(abs, v) for v in thirdq)
    end

    @testset "$(rpad("a mobility without ∂M/∂u, and a wrong SHAPE, are rejected",76))" begin
        s = TensorSplineSpace((4, 4), 2)
        N = nbasis(s)
        φ̂ = project(s, p -> sin(p[1]) * cos(p[2]))

        @test_throws DimensionMismatch CollisionBracket(s, zeros(N - 1))
        @test_throws DimensionMismatch CollisionBracket(s, zeros(N, N - 1))
        @test_throws DimensionMismatch CollisionBracket(s, φ̂; density = ones(3))
        # a function mobility is not differenced silently
        @test_throws ArgumentError CollisionBracket(s, φ̂; mobility = (x, u) -> u)
        @test CollisionBracket(s, φ̂; mobility = (x, u) -> u,
            mobility_derivative = 1) isa MetricBracket
        # the bracket is two-dimensional by definition: the ⊥ that carries its degeneracy
        # has no meaning on three axes, so there is no method rather than a chosen plane
        @test_throws MethodError CollisionBracket(TensorSplineSpace((4, 4, 4), 2), zeros(64))
    end

    @testset "$(rpad("on a MAPPED domain the gradients are the physical ones",76))" begin
        # The annulus: the polar chart with the pole cut out, so a tensor-product space
        # carries it and the physical gradient has a closed form to check against.
        R₀, R₁ = 0.4, 1.3
        F(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
        DF(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]
        s = TensorSplineSpace((UniformMesh(4, R₀ .. R₁), UniformMesh(8, 0 .. 2π)), 2,
            (Dirichlet(), Periodic()))
        pb = PulledBack(s, F, DF)
        N = nbasis(s)
        φ̂, û = randn(N), randn(N)
        mob(x, u) = 0.7 + 0.3 * x[1]^2

        b = CollisionBracket(s, φ̂, pb; mobility = mob, mobility_derivative = 0)

        # The O(N_q²) double sum of the definition, with ∂ₓ = cos θ ∂_r − (sin θ/r) ∂_θ and
        # ∂_y = sin θ ∂_r + (cos θ/r) ∂_θ written out here rather than taken from `PulledBack`.
        x̂ = quadrature_nodes(s)
        w = quadrature_weights(s)
        θ = [pt[2] for pt in x̂]
        r = [pt[1] for pt in x̂]
        P̂ = (Matrix(basis_values(s, (1, 0))), Matrix(basis_values(s, (0, 1))))
        Px = P̂[1] .* cos.(θ)' .- P̂[2] .* (sin.(θ) ./ r)'
        Py = P̂[1] .* sin.(θ)' .+ P̂[2] .* (cos.(θ) ./ r)'
        c = [w[q] * r[q] * mob(F(x̂[q]), 0.0) for q in eachindex(w)]
        gφ = (Px' * φ̂, Py' * φ̂)

        Aref = zeros(N, N)
        for q in eachindex(w), q′ in eachindex(w)

            z = (gφ[1][q] - gφ[1][q′], gφ[2][q] - gφ[2][q′])
            v = (-z[2]) .* (Px[:, q] .- Px[:, q′]) .+ z[1] .* (Py[:, q] .- Py[:, q′])
            Aref .+= (0.5 * c[q] * c[q′]) .* (v * v')
        end

        A = metric_operator(b, û)
        @test maximum(abs, A .- Aref) / maximum(abs, Aref) < 1e-10

        # CONTROL: the frame dropped and nothing else. The mobility is composed with the map
        # so that the kernel weights are identical and the frame is the only difference.
        bf = CollisionBracket(s, φ̂; density = measure(pb),
            mobility = (p, u) -> mob(F(p), u), mobility_derivative = 0)
        @test maximum(abs, metric_operator(bf, û) .- Aref) / maximum(abs, Aref) > 0.1

        # CONTROL: the frame transposed. `|det J|` is untouched by a transpose, so the
        # measure, the pairing and every area agree and only the direction is wrong.
        pbᵀ = PulledBack(s, F, x -> Matrix(DF(x)'))
        @test measure(pbᵀ) ≈ measure(pb)
        bᵀ = CollisionBracket(s, φ̂, pbᵀ; mobility = mob, mobility_derivative = 0)
        @test maximum(abs, metric_operator(bᵀ, û) .- Aref) / maximum(abs, Aref) > 0.1

        # Neither control is visible to the structural checks: they are algebraic properties
        # of Q₂ and say nothing about which gradient was handed in.
        for bad in (b, bf, bᵀ)
            @test issymmetric(bad, û)
            @test ispositive_semidefinite(bad, û)
            @test degeneracy_residual(bad, û) < 1e-11
        end

        # A mobility is a function of the physical point once a pullback is supplied.
        bx = CollisionBracket(s, φ̂, pb; mobility = (x, u) -> x[1], mobility_derivative = 0)
        @test GeometricBrackets._collision_state(bx, û).M ≈ [x[1] for x in nodes(pb)]
    end

    @testset "$(rpad("an IDENTITY pullback reproduces the unmapped bracket",76))" begin
        s = TensorSplineSpace((UniformMesh(4, 1.0 .. 2.0), UniformMesh(4, 0.0 .. 1.0)), 2,
            (Dirichlet(), Dirichlet()))
        N = nbasis(s)
        ρ(x) = 1 / x[1]
        pb = PulledBack(s, x -> (x[1], x[2]), _ -> [1.0 0.0; 0.0 1.0]; density = ρ)
        û = randn(N)
        Λ = Matrix{Float64}(I, N, N)

        plain = CollisionBracket(s, Λ; density = ρ, mobility = (x, u) -> 1 + u^2,
            mobility_derivative = (x, u) -> 2u)
        mapped = CollisionBracket(s, Λ, pb; mobility = (x, u) -> 1 + u^2,
            mobility_derivative = (x, u) -> 2u)

        G = metric_matrix(plain, û)
        @test maximum(abs, metric_matrix(mapped, û) .- G) / maximum(abs, G) < 1e-12

        v = randn(N)
        @test metric_apply(mapped, û, v) ≈ metric_apply(plain, û, v)
        @test metric_directional(mapped, û, v) ≈
              metric_directional(plain, û, v)

        # The pairing an identity map builds is the space's own mass matrix.
        @test Matrix(GeometricBrackets.weighted_matrix(s, volume_element(pb), (0, 0), (
            0, 0))) ≈
              Matrix(mass_matrix(s))
    end
end
