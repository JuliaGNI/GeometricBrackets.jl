using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, RandomMesh
using Test

# Camassa-Holm is a PROTOTYPE: it accompanies no manuscript. These tests assert the same
# structural properties as for KdV, and nothing beyond them.

@testset "$(rpad("Camassa-Holm Tests (prototype)",80))" begin
    @testset "$(rpad("Helmholtz map between velocity and momentum",76))" begin
        s = SplineSpace(UniformMesh(32, 2π), 3)
        sys = CamassaHolmSystem(s)
        û = project(s, x -> 1 + 0.5sin(x))
        m̂ = momentum(sys, û)
        @test velocity(sys, m̂) ≈ û                       # the two maps are inverse
        # m = u - u_xx: for u = 1 + a sin x the momentum is 1 + 2a sin x
        xs = collect(range(0, 2π, length = 101))
        @test maximum(abs, evaluate(s, m̂, xs) .- (1 .+ 2 .* 0.5 .* sin.(xs))) < 1e-5
    end

    @testset "$(rpad("brackets are antisymmetric; the constant one is Poisson",76))" begin
        for N in (12, 16, 20)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            sys = CamassaHolmSystem(s)
            m̂ = 0.3 .* randn(N)
            P1 = poisson_matrix(sys.bracket1, m̂)
            P2 = poisson_matrix(sys.bracket2, m̂)
            @test maximum(abs, P1 + P1') < 1e-10 * maximum(abs, P1)
            @test maximum(abs, P2 + P2') < 1e-10 * maximum(abs, P2)
            @test jacobi_residual(sys.bracket1, m̂) == 0
            @test jacobi_residual(sys.bracket2, m̂) > 0.1      # Virasoro again: not Jacobi
        end
        @test_throws ArgumentError camassa_holm_bracket_1(SplineSpace(16, 1))
    end

    @testset "$(rpad("BI-HAMILTONIAN: the two flows agree, converging at about 2p",76))" begin
        # The point of the pairing, and what caught the variable being wrong: the
        # variational derivatives are with respect to the MOMENTUM, not the velocity. Taken
        # with respect to u the two flows differ by 44% and do not converge at all.
        for p in (3, 4)
            errs = Float64[]
            for N in (32, 64)
                s = SplineSpace(UniformMesh(N, 2π), p)
                sys = CamassaHolmSystem(s)
                m̂ = momentum(sys, project(s, x -> 1 + 0.5sin(x)))
                f1 = vectorfield(sys.flow1, m̂)
                f2 = vectorfield(sys.flow2, m̂)
                push!(errs, maximum(abs, f1 .- f2) / maximum(abs, f1))
            end
            @test errs[1] < 1e-5
            @test errs[2] < errs[1]
        end
    end

    @testset "$(rpad("semi-discrete conservation on ANY mesh",76))" begin
        for mk in (n -> UniformMesh(n, 2π), n -> RandomMesh(n, 2π))
            s = SplineSpace(mk(16), 3)
            sys = CamassaHolmSystem(s)
            m̂ = momentum(sys, project(s, x -> 1 + 0.5sin(x)))
            @test abs(dot(gradient(sys.H2, s, m̂), vectorfield(sys.flow1, m̂))) < 1e-10
            @test abs(dot(gradient(sys.H1, s, m̂), vectorfield(sys.flow2, m̂))) < 1e-10
            # The mass of the momentum is Casimir-strength under the CONSTANT bracket,
            # where P¹ g = 0 identically, and only approximately conserved under the
            # momentum-dependent one. That is not a defect of the assembly: under P² its
            # conservation rests on ∫ u_x m dx = 0, which is a statement about how well the
            # discretisation resolves two total derivatives, not an identity in m̂. KdV is
            # the more fortunate case, not the general one.
            g = basis_integrals(s)
            @test maximum(abs, poisson_apply(sys.bracket1, m̂, g)) < 1e-10
            @test abs(dot(g, vectorfield(sys.flow1, m̂))) < 1e-10
            @test abs(dot(g, vectorfield(sys.flow2, m̂))) < 1e-6
        end
    end

    @testset "$(rpad("gradients against finite differences",76))" begin
        s = SplineSpace(UniformMesh(12, 2π), 3)
        sys = CamassaHolmSystem(s)
        m̂ = 1 .+ 0.2 .* randn(12)
        for H in (sys.H1, sys.H2)
            g = gradient(H, s, m̂)
            h = 1e-6
            fd = [(hamiltonian(H, s, m̂ .+ h .* e) - hamiltonian(H, s, m̂ .- h .* e)) / 2h
                  for e in eachcol(Matrix(I, 12, 12))]
            @test g ≈ fd atol = 1e-5 * max(1, maximum(abs, g))
        end
    end

    @testset "$(rpad("time integration keeps the generating Hamiltonian",76))" begin
        s = SplineSpace(UniformMesh(24, 2π), 3)
        sys = CamassaHolmSystem(s)
        m0 = momentum(sys, project(s, x -> 1 + 0.5sin(x)))

        mid = integrate(sys, Integrator(sys.flow2, ImplicitMidpoint(), 1e-3), m0, 300;
            stride = 10)
        @test drift(mid, :H1) < 1e-12          # H₁ is quadratic in m̂ and midpoint symplectic
        @test absolute_drift(mid, :C0) < 1e-12

        for meth in (AverageVectorField(), Gonzalez())
            traj = integrate(sys, Integrator(sys.flow1, meth, 1e-3), m0, 300; stride = 10)
            @test drift(traj, :H2) < 1e-10     # the cubic one, held by an energy method
            @test absolute_drift(traj, :C0) < 1e-10
        end
    end

    @testset "$(rpad("peakon is a weak solution with a corner",76))" begin
        L = 2π
        u = peakon(1.0, π, L)
        @test u(π) ≈ 1.0 atol = 1e-2
        @test u(0.0) ≈ u(L) atol = 1e-10
        # the corner: the one-sided slopes at the crest differ in sign
        h = 1e-5
        @test (u(π + h) - u(π)) / h < 0
        @test (u(π) - u(π - h)) / h > 0
    end
end
