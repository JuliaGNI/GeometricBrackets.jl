using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh
using Test

include("helpers/meshes.jl")

Random.seed!(0x5c1e9a3b)

@testset "$(rpad("Hamiltonian Flow Tests",80))" begin
    @testset "$(rpad("vectorfield is the bracket contracted with the gradient",76))" begin
        s = SplineSpace(12, 3)
        sys = KdVSystem(s)
        û = 0.3 .* randn(12)
        @test vectorfield(sys.flow1, û) ≈
              poisson_matrix(sys.bracket1, û) * gradient(sys.H1, s, û)
        @test vectorfield(sys.flow2, û) ≈
              poisson_matrix(sys.bracket2, û) * gradient(sys.H2, s, û)
        du = similar(û)
        @test vectorfield!(du, sys.flow1, û) ≈ vectorfield(sys.flow1, û)
    end

    @testset "$(rpad("Jacobian against finite differences",76))" begin
        for (nm, mk) in SPLINE_MESHES
            s = SplineSpace(mk(12), 3)
            sys = KdVSystem(s)
            û = 0.3 .* randn(12)
            for flow in (sys.flow1, sys.flow2)
                J = jacobian(flow, û)
                h = 1e-6
                for m in (1, 5, 11)
                    e = zeros(12)
                    e[m] = h
                    fd = (vectorfield(flow, û .+ e) .- vectorfield(flow, û .- e)) ./ 2h
                    @test J[:, m] ≈ fd atol = 1e-5 * max(1, maximum(abs, fd))
                end
            end
        end
    end

    @testset "$(rpad("the generating Hamiltonian is conserved on ANY mesh at ANY nq",76))" begin
        # dH/dt = ∇Hᵀ P ∇H = 0 by antisymmetry alone. No property of the quadrature is
        # needed -- this is exactly what the skew-symmetrised assembly buys, and it is why
        # nq = 2 below is not a mistake.
        for (nm, mk) in SPLINE_MESHES, nq in (2, 3, 5, 8)

            s = SplineSpace(mk(16), 3; nq = nq)
            sys = KdVSystem(s)
            û = 0.3 .* randn(16)
            @test abs(dot(gradient(sys.H1, s, û), vectorfield(sys.flow1, û))) < 1e-11
            @test abs(dot(gradient(sys.H2, s, û), vectorfield(sys.flow2, û))) < 1e-11
        end
    end

    @testset "$(rpad("the mass is conserved by both flows",76))" begin
        for (nm, mk) in SPLINE_MESHES
            s = SplineSpace(mk(16), 3)
            sys = KdVSystem(s)
            g = basis_integrals(s)
            û = 0.3 .* randn(16)
            @test abs(dot(g, vectorfield(sys.flow1, û))) < 1e-11
            @test abs(dot(g, vectorfield(sys.flow2, û))) < 1e-11
        end
    end

    @testset "$(rpad("the mass is Casimir-strength under P1, Hamiltonian-specific under P2",76))" begin
        s = SplineSpace(UniformMesh(16, 2π), 3)
        sys = KdVSystem(s)
        g = basis_integrals(s)
        û = 0.3 .* randn(16)
        # P¹ g = 0 identically: any increment in the range of P¹ preserves the mass
        @test maximum(abs, poisson_apply(sys.bracket1, û, g)) < 1e-12
        # P² g ≠ 0; what vanishes is gᵀ P² ∇H₂, pointwise in û
        @test maximum(abs, poisson_apply(sys.bracket2, û, g)) > 1e-6
        @test abs(dot(g, vectorfield(sys.flow2, û))) < 1e-11
    end

    @testset "$(rpad("the two flows differ, and by the projection error",76))" begin
        # They are not the same scheme: the first inserts an L² projection between its two
        # derivatives. The difference converges at O(h^{2p}) -- the same order as either
        # scheme's own consistency error -- so it is structural, not a matter of accuracy.
        errs = Float64[]
        for N in (16, 32, 64)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            sys = KdVSystem(s)
            û = project(s, sin)
            d = vectorfield(sys.flow1, û) .- vectorfield(sys.flow2, û)
            push!(errs, maximum(abs, d))
        end
        @test all(>(0), errs)                                   # they really do differ
        rate = log2(errs[end - 1] / errs[end])
        @test rate > 4                                          # and converge fast
    end

    @testset "$(rpad("Magri rung: P2 g = -2 P1 dH2/du, exact at finite N",76))" begin
        # The transport part of P² carries the sign of the convention while ∂ₓ³ does not, so
        # this rung is -2 where the older u_t = 6uu_x - u_xxx convention had +2. In the
        # continuum: P²·1 = -2u_x against 2 P¹ u = 2u_x.
        for N in (12, 16, 20)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            sys = KdVSystem(s)
            û = 0.3 .* randn(N)
            lhs = poisson_apply(sys.bracket2, û, basis_integrals(s))
            rhs = -2 .* poisson_apply(sys.bracket1, û, gradient(sys.H2, s, û))
            @test lhs ≈ rhs atol = 1e-10 * maximum(abs, rhs)
        end
    end
end
