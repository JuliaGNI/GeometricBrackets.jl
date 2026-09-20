using GeometricBrackets
using LinearAlgebra
using SimpleSplines: UniformMesh
using Test

@testset "$(rpad("Diagnostics Tests",80))" begin
    s = SplineSpace(UniformMesh(16, 2π), 3)
    sys = KdVSystem(s)
    u0 = project(s, sin)

    @testset "$(rpad("fused invariants agree with the individual Hamiltonians",76))" begin
        û = 0.3 .* randn(16)
        inv = invariants(sys, û)
        @test inv.H1 ≈ hamiltonian(sys.H1, s, û)
        @test inv.H2 ≈ hamiltonian(sys.H2, s, û)
        @test inv.C0 ≈ hamiltonian(sys.C0, s, û)
        @test invariant_names(sys) == (:H1, :H2, :C0)
        @test keys(inv) == invariant_names(sys)
    end

    @testset "$(rpad("trajectory shape and windowing",76))" begin
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
        traj = integrate(sys, integ, u0, 100; stride = 10)
        @test length(traj) == 10
        @test size(traj.deviation) == (3, 10)
        @test traj.t ≈ (1:10) .* (10 * 1e-3)
        @test all(≥(0), traj.deviation)
        @test length(traj.final) == 16
        # the initial state is not modified
        @test u0 ≈ project(s, sin)
    end

    @testset "$(rpad("windowed maxima dominate a plain sample",76))" begin
        # The whole reason for windowing. The error envelope of a geometric integrator
        # oscillates with a period of tens of steps, so the value at the END of each window
        # -- what plain subsampling would record -- can sit well below the window's peak.
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 2e-3)
        fine = integrate(sys, integ, u0, 200; stride = 1)
        per_step = deviation(fine, :H1)

        windowed = integrate(sys, Integrator(sys.flow1, ImplicitMidpoint(), 2e-3), u0, 200;
            stride = 20)
        sampled = per_step[20:20:200]                    # what subsampling would report

        @test length(deviation(windowed, :H1)) == length(sampled)
        @test all(deviation(windowed, :H1) .≥ sampled .- 1e-15)
        # and it is strictly larger somewhere: the two are not the same series
        @test maximum(deviation(windowed, :H1) .- sampled) > 0
    end

    @testset "$(rpad("drift, absolute_drift and growth",76))" begin
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
        traj = integrate(sys, integ, u0, 200; stride = 10)
        @test drift(traj, :H1) ≥ 0
        @test absolute_drift(traj, :H1) ≈ drift(traj, :H1) * abs(traj.reference.H1)
        @test growth(traj, :H1) > 0
        @test_throws ArgumentError deviation(traj, :nope)
    end

    @testset "$(rpad("a zero reference value forbids a relative drift",76))" begin
        # the mass of the cosine initial condition is exactly zero; dividing by it would
        # report Inf for a quantity that is in fact conserved to round-off
        c0 = project(s, cosine(2π))
        traj = integrate(sys, Integrator(sys.flow1, ImplicitMidpoint(), 1e-3), c0, 100;
            stride = 10)
        @test abs(traj.reference.C0) < 1e-14
        @test_throws ArgumentError drift(traj, :C0)
        @test absolute_drift(traj, :C0) < 1e-12
    end

    @testset "$(rpad("argument checks",76))" begin
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
        @test_throws ArgumentError integrate(sys, integ, u0, 10; stride = 0)
        @test_throws ArgumentError integrate(sys, integ, u0, 5; stride = 10)
    end
end
