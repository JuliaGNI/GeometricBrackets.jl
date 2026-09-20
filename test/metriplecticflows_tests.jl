using PoissonBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh
using Test

# A constant, NEGATIVE definite "metric" bracket. It is not a metric bracket at all — that is
# the point. The sign assertions below are only worth making if something can fail them, and
# nothing built by this package can: `ispositive_semidefinite` holds by construction for all
# three, so the control has to be written by hand.
struct NegativeMetricBracket{T} <: MetricBracket{T}
    G::Matrix{T}
end

PoissonBrackets.metric_matrix(b::NegativeMetricBracket, û::AbstractVector) = b.G

function PoissonBrackets.metric_derivative(b::NegativeMetricBracket{T},
        û::AbstractVector) where {T}
    N = length(û)
    zeros(T, N, N, N)
end

# `metric_directional` written out from its definition: the full O(N³) tensor, contracted by
# hand. This is what the specialised paths replace, and comparing against it is the only way
# to tell a faster derivative from a wrong one.
function contracted_derivative(b::MetricBracket, û::AbstractVector, v::AbstractVector)
    dG = metric_derivative(b, û)
    N = length(û)
    [sum(dG[m, i, j] * v[j] for j in 1:N) for i in 1:N, m in 1:N]
end

# The central-difference Jacobian of the vector field, as a test oracle only. It is banned
# from the production path — see `jacobian(::MetriplecticFlow, û)` — but its ~1e-10 floor is
# three orders below what a wrong sign or a dropped term costs, so it settles this question.
function finite_difference_jacobian(f, û::AbstractVector, ε::Real)
    N = length(û)
    J = zeros(eltype(û), N, N)
    for m in 1:N
        e = zeros(N)
        e[m] = ε
        J[:, m] = (vectorfield(f, û .+ e) .- vectorfield(f, û .- e)) ./ 2ε
    end
    return J
end

# The 1D projector-bracket flow: δH/δu = u, so Λ = 𝕀 and H = ½∫u² generates the bracket that
# then annihilates it. S is the Dirichlet energy, and the field is a nonlinear heat equation.
function projector_flow(N = 16)
    s = SplineSpace(UniformMesh(N, 2π), 3)
    Λ = Matrix{Float64}(I, N, N)
    (s, ProjectorBracket(s, Λ),
        QuadraticHamiltonian(Matrix(mass_matrix(s))),
        QuadraticHamiltonian(Matrix(stiffness_matrix(s))))
end

# The 2D case: the screened Poisson solve ψ̂ = (𝕂 + ½𝕄)⁻¹𝕄 ω̂ as the generating map, so
# ∂H/∂û = 𝕄Λû and 𝕄Λ is symmetric — Λ really is 𝕄⁻¹∂H/∂û of a Hamiltonian. S = ½∫u².
function tensor_flow_parts(n = 5)
    s = TensorSplineSpace((n, n), 2)
    M = Matrix(mass_matrix(s))
    Λ = (Matrix(stiffness_matrix(s)) + 0.5M) \ M
    (s, Λ, QuadraticHamiltonian(M * Λ), QuadraticHamiltonian(M))
end

@testset "$(rpad("Metriplectic Flow Tests",80))" begin
    rng = MersenneTwister(0x2f8a41c7)

    @testset "$(rpad("both flows are an ABSTRACTFLOW, and the bracket is optional",76))" begin
        s, b, H, S = projector_flow()
        f = MetriplecticFlow(s, b, H, S)

        @test f isa AbstractFlow{Float64}
        @test eltype(f) == Float64
        @test length(f) == nbasis(s)
        @test f.bracket === nothing                 # the four-argument form drops it
        @test PoissonBrackets.metric(f) === b
        @test PoissonBrackets.entropy(f) === S
        @test PoissonBrackets.hamiltonian(f) === H

        # the generalisation is what lets a metriplectic field reach the integrators at all
        @test HamiltonianFlow(s, kdv_bracket_1(s), H) isa AbstractFlow{Float64}
        @test MetriplecticFlow(s, kdv_bracket_1(s), b, H, S) isa AbstractFlow{Float64}

        û = 0.3 .* randn(rng, nbasis(s))
        @test PoissonBrackets.entropy(f, û) ≈ hamiltonian(S, s, û)
        @test PoissonBrackets.entropy_gradient(f, û) ≈ gradient(S, s, û)
        @test PoissonBrackets.entropy_hessian(f, û) ≈ hessian(S, s, û)
        # the Hamiltonian half keeps the names the integrators call
        @test PoissonBrackets.hamiltonian(f, û) ≈ hamiltonian(H, s, û)
        @test gradient(f, û) ≈ gradient(H, s, û)
    end

    @testset "$(rpad("bracket = nothing is the PURE dissipative field",76))" begin
        s, b, H, S = projector_flow()
        û = project(s, x -> sin(x) + 0.3cos(2x)^2)

        f = MetriplecticFlow(s, b, H, S)
        g = gradient(S, s, û)
        @test vectorfield(f, û) ≈ -metric_apply(b, û, g)
        @test vectorfield(f, û) ≈ -metric_matrix(b, û) * g
        du = similar(û)
        @test vectorfield!(du, f, û) ≈ vectorfield(f, û)

        # and adding the Poisson half adds exactly `poisson_apply` and nothing else
        p = kdv_bracket_1(s)
        fp = MetriplecticFlow(s, p, b, H, S)
        @test vectorfield(fp, û) ≈
              vectorfield(f, û) .+ poisson_apply(p, û, gradient(H, s, û))
        # the two really are different fields -- the Poisson half is not a rounding error
        @test maximum(abs, vectorfield(fp, û) .- vectorfield(f, û)) >
              0.1 * maximum(abs, vectorfield(f, û))

        # `sin` alone would make this testset vacuous: it is an eigenfunction of -∂ₓ², so ∂S/∂û
        # is parallel to ∂H/∂û and lands in the kernel of 𝔾
        @test maximum(abs, vectorfield(f, project(s, sin))) < 1e-12
        @test maximum(abs, vectorfield(f, û)) > 1
    end

    @testset "$(rpad("the Jacobian is ANALYTIC and matches a central difference",76))" begin
        s, b, H, S = projector_flow()
        N = nbasis(s)
        û = 0.3 .* randn(rng, N)

        for f in (MetriplecticFlow(s, b, H, S), MetriplecticFlow(
            s, kdv_bracket_1(s), b, H, S))
            J = jacobian(f, û)
            fd = finite_difference_jacobian(f, û, 1e-6)
            @test maximum(abs, J - fd) < 1e-7 * maximum(abs, J)
            # the control: the floor is the difference scheme's, not the derivative's, so a
            # Jacobian wrong by a per cent -- far less than any sign or index error -- fails
            @test maximum(abs, 1.01 * J - fd) > 1e-7 * maximum(abs, J)
        end
    end

    @testset "$(rpad("the Jacobian is analytic in TWO dimensions too",76))" begin
        s, Λ, H, S = tensor_flow_parts()
        N = nbasis(s)
        û = 0.3 .* randn(rng, N)

        for b in (DoubleBracket(s, Λ), CollisionBracket(s, Λ),
            CollisionBracket(s, Λ; mobility = (x, u) -> 1 + u^2,
            mobility_derivative = (x, u) -> 2u))
            f = MetriplecticFlow(s, b, H, S)
            J = jacobian(f, û)
            fd = finite_difference_jacobian(f, û, 1e-6)
            @test maximum(abs, J - fd) < 1e-6 * maximum(abs, J)
            @test maximum(abs, 1.01 * J - fd) > 1e-6 * maximum(abs, J)
        end
    end

    @testset "$(rpad("metric_directional is the FULL tensor, contracted",76))" begin
        s, Λ, _, _ = tensor_flow_parts()
        N = nbasis(s)
        û = 0.3 .* randn(rng, N)
        v = randn(rng, N)

        # every Λ-generated bracket, against the O(N³) definition on a problem small enough
        # to afford it
        for b in (DoubleBracket(s, Λ), ProjectorBracket(s, Λ), CollisionBracket(s, Λ),
            CollisionBracket(s, Λ; mobility = (x, u) -> 1 + u^2,
            mobility_derivative = (x, u) -> 2u))
            D = metric_directional(b, û, v)
            Dfull = contracted_derivative(b, û, v)
            @test size(D) == (N, N)
            @test maximum(abs, D - Dfull) < 1e-12 * maximum(abs, Dfull)
            @test maximum(abs, Dfull) > 1e-3      # not zero, so the agreement has content
        end

        # a prescribed generating field makes the bracket constant, and the contraction is
        # then identically zero without any assembly
        ĥ = project(s, p -> cos(p[1])^2 * sin(p[2])^2)
        for b in (DoubleBracket(s, ĥ), ProjectorBracket(s, ĥ), CollisionBracket(s, ĥ))
            @test metric_directional(b, û, v) == zeros(N, N)
        end

        # the generic fallback -- a bracket that defines only the three interface methods --
        # still answers, by forming the tensor
        G = Matrix(mass_matrix(s))
        nb = NegativeMetricBracket(-G)
        @test metric_directional(nb, û, v) == zeros(N, N)
    end

    @testset "$(rpad("H is conserved to ROUND-OFF while S falls MONOTONICALLY",76))" begin
        s, b, H, S = projector_flow()
        f = MetriplecticFlow(s, b, H, S)
        u0 = project(s, x -> sin(x) + 0.3cos(2x)^2)

        û = copy(u0)
        integ = Integrator(f, ImplicitMidpoint(), 1e-3; û₀ = u0)
        H0 = hamiltonian(H, s, u0)
        Sseries = [hamiltonian(S, s, û)]
        production = [entropy_production(f, û)]
        dH = 0.0
        for _ in 1:50
            integrate_step!(û, integ)
            dH = max(dH, abs(hamiltonian(H, s, û) - H0))
            push!(Sseries, hamiltonian(S, s, û))
            push!(production, entropy_production(f, û))
        end

        # H is conserved by the bracket, not by the method: 𝔾 ∂H/∂û = 0 exactly, and the
        # midpoint increment lies in the range of 𝔾(ū)
        @test dH / abs(H0) < 1e-14
        # S falls at every step, and by much more than the drift in H
        @test all(<(0), diff(Sseries))
        @test Sseries[1] - Sseries[end] > 0.1
        @test all(>(0), production)

        # and the production really is -Ṡ. Against the *trapezoid* of the production over the
        # step, not against its value at the left end: a forward difference of S is only first
        # order, and misses by 1.4% at Δt = 1e-3 and 0.71% at 5e-4, which measures the
        # difference scheme rather than the identity. The trapezoid residual is 4.7e-5 and
        # 1.2e-5, i.e. second order, so the ratio below is the claim and the threshold is not.
        function step_residual(Δt)
            v = copy(u0)
            one_step = Integrator(f, ImplicitMidpoint(), Δt; û₀ = u0)
            p0 = entropy_production(f, v)
            S0 = hamiltonian(S, s, v)
            integrate_step!(v, one_step)
            p1 = entropy_production(f, v)
            abs(-(hamiltonian(S, s, v) - S0) / Δt - (p0 + p1) / 2) / p0
        end
        coarse, fine = step_residual(1e-3), step_residual(5e-4)
        @test coarse < 1e-4
        @test coarse / fine > 3.5                        # O(Δt²), against a predicted 4

        # RK4 covers the explicit case. It holds H to its own order rather than exactly --
        # the degeneracy is a property of the bracket, but exact conservation of a quadratic
        # first integral is a property of the midpoint rule, and RK4 has neither.
        ûe = copy(u0)
        rk = Integrator(f, RungeKutta4(), 1e-3)
        dHe = 0.0
        Se = [hamiltonian(S, s, ûe)]
        for _ in 1:50
            integrate_step!(ûe, rk)
            dHe = max(dHe, abs(hamiltonian(H, s, ûe) - H0))
            push!(Se, hamiltonian(S, s, ûe))
        end
        @test all(<(0), diff(Se))
        # 1.6e-11 against the midpoint rule's 5.3e-16: four orders, and neither is the
        # other's round-off
        @test dHe / abs(H0) > 100 * dH / abs(H0)
        @test dHe / abs(H0) < 1e-8
        # both integrate the same field, and after 50 steps they agree to 1.2e-6 -- which is
        # the midpoint rule's own O(Δt²) error and not a disagreement about the flow
        @test maximum(abs, ûe .- û) < 1e-5 * maximum(abs, û)
    end

    @testset "$(rpad("entropy_production is SIGN-definite, and the sign has content",76))" begin
        s, b, H, S = projector_flow()
        N = nbasis(s)
        û = 0.3 .* randn(rng, N)

        f = MetriplecticFlow(s, b, H, S)
        g = gradient(S, s, û)
        @test entropy_production(f, û) > 0
        @test entropy_production(f, û) ≈ dot(g, metric_matrix(b, û) * g)
        # 𝔾 is degenerate on ∂H/∂û, so a flow whose entropy IS the Hamiltonian produces none
        @test entropy_production(MetriplecticFlow(s, b, H, H), û) <
              1e-14 * entropy_production(f, û)

        # the control: a bracket that is not positive semi-definite produces entropy of the
        # wrong sign, which is a backward heat equation and not a small error. Nothing this
        # package assembles can do it, which is why the control is written by hand.
        nb = NegativeMetricBracket(-Matrix(mass_matrix(s)))
        @test entropy_production(MetriplecticFlow(s, nb, H, S), û) < 0
    end

    @testset "$(rpad("degeneracy_residual against the FLOW's own Hamiltonian",76))" begin
        s, b, H, S = projector_flow()
        û = project(s, x -> sin(x) + 0.3cos(2x)^2)

        # `b` is generated by Λ = 𝕀, i.e. by ∂H/∂û = 𝕄û, so H = ½ûᵀ𝕄û is the Hamiltonian it
        # is degenerate on. A flow built on that pair conserves H to round-off.
        good = MetriplecticFlow(s, b, H, S)
        @test degeneracy_residual(good, û) < 1e-13
        @test degeneracy_residual(good, û) ≈ degeneracy_residual(b, û, gradient(H, s, û))

        # Hand the same bracket a DIFFERENT Hamiltonian -- the Dirichlet energy, whose
        # gradient 𝕂û is not the 𝕄Λû the bracket annihilates -- and the flow loses it at
        # order one. The two-argument form cannot see that: it takes the gradient from the
        # bracket, so it still reports the bracket as clean. That is the whole reason the
        # flow method exists.
        Hbad = QuadraticHamiltonian(Matrix(stiffness_matrix(s)))
        Sbad = QuadraticHamiltonian(Matrix(stiffness_matrix(s)) + Matrix(mass_matrix(s)))
        bad = MetriplecticFlow(s, b, Hbad, Sbad)
        @test degeneracy_residual(b, û) < 1e-13
        @test degeneracy_residual(bad, û) > 0.1

        function drift(f, E)
            v = copy(û)
            integ = Integrator(f, ImplicitMidpoint(), 1e-3; û₀ = û)
            E0 = hamiltonian(E, s, û)
            for _ in 1:50
                integrate_step!(v, integ)
            end
            abs(hamiltonian(E, s, v) - E0) / abs(E0)
        end
        # the mismatched flow is a real flow -- it moves -- and it does not hold its energy
        @test maximum(abs, vectorfield(bad, û)) > 1
        @test drift(good, H) < 1e-14
        @test drift(bad, Hbad) > 0.01
    end

    @testset "$(rpad("the AbstractFlow interface, and what refuses a dissipative flow",76))" begin
        s, b, H, S = projector_flow()
        f = MetriplecticFlow(s, b, H, S)
        û = project(s, x -> sin(x) + 0.3cos(2x)^2)

        # `space` is the third method of the interface: `Integrator` needs it before any
        # vector field is evaluated. Unqualified, because a method a downstream flow is
        # required to define has to be reachable through `using PoissonBrackets`
        @test space(f) === s
        @test nbasis(space(f)) == length(f)

        # `:mixed` refuses at CONSTRUCTION, the discrete gradients only at the first step
        @test_throws MethodError Integrator(f, ImplicitMidpoint(), 1e-3; formulation = :mixed)
        for m in (Gonzalez(), GonzalezMass())
            integ = Integrator(f, m, 1e-3)          # the constructor itself is happy
            @test_throws MethodError integrate_step!(copy(û), integ)
        end

        # `poisson_defect` names the reason rather than dying inside `poisson_matrix`
        @test_throws ArgumentError poisson_defect(Integrator(f, ImplicitMidpoint(), 1e-3), û)
        withbracket = MetriplecticFlow(s, kdv_bracket_1(s), b, H, S)
        @test_throws ArgumentError poisson_defect(
            Integrator(withbracket, ImplicitMidpoint(), 1e-3), û)
    end
end
