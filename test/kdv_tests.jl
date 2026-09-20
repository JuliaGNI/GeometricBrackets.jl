using GeometricBrackets
using LinearAlgebra
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh
using Test

@testset "$(rpad("KdV Tests",80))" begin
    @testset "$(rpad("both flows reproduce -6 u u_x - u_xxx",76))" begin
        # for u = sin x the right-hand side is -3 sin 2x + cos x: only the nonlinear term
        # flips with the convention, the -u_xxx being common to both
        for p in 2:4
            errs = Float64[]
            for N in (32, 64)
                s = SplineSpace(UniformMesh(N, 2π), p)
                sys = KdVSystem(s)
                û = project(s, sin)
                ex = project(s, x -> -3sin(2x) + cos(x))
                xs = collect(range(0, 2π, length = 201))
                sc = maximum(abs, evaluate(s, ex, xs))
                e1 = maximum(abs, evaluate(s, vectorfield(sys.flow1, û) .- ex, xs)) / sc
                e2 = maximum(abs, evaluate(s, vectorfield(sys.flow2, û) .- ex, xs)) / sc
                @test e1 < 1e-3
                @test e2 < 1e-3
                push!(errs, e1)
            end
            # The observed rate is about 2p. At p = 4 the error has already reached the
            # round-off floor of the assembly by N = 32 (1.6e-11 falling only to 5.4e-12),
            # so the measured rate there says nothing about the scheme; the floor itself is
            # the assertion worth making.
            if p ≤ 3
                @test log2(errs[1] / errs[2]) > 2p - 1
            else
                @test errs[2] < 1e-10
            end
        end
    end

    @testset "$(rpad("the sign convention: solitons are elevations",76))" begin
        # u_t + 6uu_x + u_xxx = 0 is the textbook convention, so the sech² solitons are
        # POSITIVE. Getting the sign wrong gives a profile that steepens and blows up
        # instead of translating.
        L = 40.0
        u = soliton(0.5, 20.0, L)
        @test u(20.0) > 0
        @test u(20.0) ≈ 2 * 0.5^2 atol = 1e-12
        @test abs(u(0.0)) < 1e-6                       # tails below round-off in this box
        @test u(20.0 + L) ≈ u(20.0) atol = 1e-12       # periodic
    end

    @testset "$(rpad("soliton translates at speed 4 kappa^2",76))" begin
        L, κ = 40.0, 0.5
        u0 = soliton(κ, 10.0, L)
        ut = soliton(κ, 10.0, L; t = 2.0)
        @test ut(10.0 + 4κ^2 * 2.0) ≈ u0(10.0) atol = 1e-10
    end

    @testset "$(rpad("Hirota two-soliton is finite and does not overflow",76))" begin
        L = 40.0
        u = two_soliton(0.6, 0.4, 12.0, 24.0, L)
        xs = range(0, L, length = 401)
        v = u.(xs)
        @test all(isfinite, v)
        @test all(≥(-1e-10), v)                        # elevations
        @test maximum(v) > 0.1
        @test u(0.0) ≈ u(L) atol = 1e-8
    end

    @testset "$(rpad("multi-soliton superposition and cosine",76))" begin
        L = 40.0
        u = solitons((0.5, 0.4), (10.0, 25.0), L)
        @test u(10.0) ≈ 2 * 0.5^2 atol = 1e-3
        @test_throws DimensionMismatch solitons((0.5,), (10.0, 25.0), L)
        c = cosine(2π)
        @test c(0.0) ≈ 1
        @test c(π) ≈ -1
    end

    @testset "$(rpad("KdVSystem assembles",76))" begin
        sys = KdVSystem(16, 3)
        @test nbasis(sys) == 16
        @test eltype(sys) == Float64
        @test length(vectorfield(sys.flow1, project(sys.space, sin))) == 16
        @test KdVSystem(SplineSpace(16, 3)) isa KdVSystem
    end

    @testset "$(rpad("REGRESSION: the drift table of the manuscript",76))" begin
        # N = 20, p = 3, dt = 2e-3 over 500 steps. Every method holds the Hamiltonian that
        # generates its flow when it is built to, every method holds the mass, and NONE
        # holds the respective other Hamiltonian.
        N, p, dt, NT = 20, 3, 2e-3, 500
        s = SplineSpace(UniformMesh(N, 2π), p)
        sys = KdVSystem(s)
        u0 = project(s, cosine(2π))

        run(flow, meth) = integrate(sys, Integrator(flow, meth, dt), u0, NT; stride = 10)

        mid1 = run(sys.flow1, ImplicitMidpoint())
        @test drift(mid1, :H1) < 1e-4              # cubic H₁ not conserved by a symplectic map
        @test drift(mid1, :H1) > 1e-8
        @test drift(mid1, :H2) < 1e-4
        @test absolute_drift(mid1, :C0) < 1e-12    # mass exact

        avf1 = run(sys.flow1, AverageVectorField())
        @test drift(avf1, :H1) < 1e-11             # energy-exact on a constant bracket
        @test drift(avf1, :H2) > 1e-8              # but the OTHER Hamiltonian still drifts
        @test absolute_drift(avf1, :C0) < 1e-12

        for m in (Gonzalez(), GonzalezMass())
            dg = run(sys.flow1, m)
            @test drift(dg, :H1) < 1e-11           # energy-exact for ANY Hamiltonian
            @test absolute_drift(dg, :C0) < 1e-12
        end

        mid2 = run(sys.flow2, ImplicitMidpoint())
        @test drift(mid2, :H2) < 1e-12             # quadratic invariant, symplectic method
        @test drift(mid2, :H1) > 1e-4              # severe loss of the other one
        @test absolute_drift(mid2, :C0) < 1e-12

        # AVF on the SECOND flow loses H₂: its energy identity needs a CONSTANT structure
        # matrix, and the average of the product is not the product of the averages.
        avf2 = run(sys.flow2, AverageVectorField())
        @test drift(avf2, :H2) > 1e-9
        @test drift(avf2, :H2) < 1e-4
        @test absolute_drift(avf2, :C0) < 1e-12
    end

    @testset "$(rpad("the mass is free, even for explicit Euler",76))" begin
        s = SplineSpace(UniformMesh(20, 2π), 3)
        sys = KdVSystem(s)
        u0 = project(s, cosine(2π))
        for meth in (ExplicitEuler(), RungeKutta4())
            traj = integrate(sys, Integrator(sys.flow1, meth, 1e-5), u0, 200; stride = 20)
            @test absolute_drift(traj, :C0) < 1e-12
        end
    end

    @testset "$(rpad("Miura map",76))" begin
        s = SplineSpace(UniformMesh(64, 2π), 3)
        v̂ = project(s, x -> 1.0 + 0.3sin(x))
        û = GeometricBrackets.miura_map(s, v̂)
        xs = collect(range(0, 2π, length = 101))
        exact = [-((1 + 0.3sin(x))^2 + 0.3cos(x)) for x in xs]
        @test maximum(abs, evaluate(s, û, xs) .- exact) < 1e-6
    end
end
