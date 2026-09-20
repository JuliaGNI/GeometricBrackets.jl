using GeometricBrackets
using LinearAlgebra
using Test

uex(x) = 2 + sin(x) + 0.3cos(2x)
dudx(x) = cos(x) - 0.6sin(2x)

@testset "$(rpad("Burgers Tests",80))" begin
    @testset "$(rpad("K is antisymmetric with the basis integrals in its kernel",76))" begin
        for (p, ne) in ((1, 25), (2, 13), (1, 24), (2, 12))
            s = LagrangeSpace(p, ne)
            N = nbasis(s)
            sys = BurgersSystem(s)
            K = sys.bracket.K
            M = mass_matrix(s)
            @test maximum(abs, K + K') < 1e-12
            @test maximum(abs, K * (M * ones(N))) < 1e-10
            @test M * ones(N) ≈ basis_integrals(s)
            @test sum(basis_integrals(s)) ≈ 2π
        end
    end

    @testset "$(rpad("an antisymmetric matrix has EVEN rank: use an odd N",76))" begin
        # For an even number of degrees of freedom the corank is two, not one, and the
        # extra kernel vector -- the sawtooth mode -- is a SPURIOUS discrete Casimir with
        # no continuum counterpart.
        for (p, ne) in ((1, 25), (2, 13), (1, 24), (2, 12))
            s = LagrangeSpace(p, ne)
            N = nbasis(s)
            K = BurgersSystem(s).bracket.K
            @test rank(K, atol = 1e-9) == (isodd(N) ? N - 1 : N - 2)
            if iseven(N)
                saw = [(-1.0)^i for i in 1:N]
                @test maximum(abs, K * (mass_matrix(s) * saw)) < 1e-10
            end
        end
    end

    @testset "$(rpad("the square-root Casimir is EXACT, not approximate",76))" begin
        for (p, ne) in ((1, 25), (2, 13))
            s = LagrangeSpace(p, ne)
            sys = BurgersSystem(s)
            u = uex.(nodes(s))
            J = poisson_matrix(sys.bracket, u)
            @test maximum(abs, J * gradient(sys.C, s, u)) < 1e-10
            # and it is a consistent quadrature of the continuous 2 ∫ √u
            xs = range(0, 2π, length = 20001)
            cont = 2 * sum(sqrt.(uex.(xs[1:(end - 1)]))) * (2π / 20000)
            @test abs(hamiltonian(sys.C, s, u) - cont) / cont < 1e-3
        end
    end

    @testset "$(rpad("consistency: J grad H reproduces 3 u u_x at order >= 2",76))" begin
        for p in 1:2
            errs = Float64[]
            hs = Float64[]
            for ne in (16, 32, 64, 128)
                s = LagrangeSpace(p, ne)
                sys = BurgersSystem(s)
                u = uex.(nodes(s))
                ud = vectorfield(sys.flow, u)
                ex = 3 .* uex.(nodes(s)) .* dudx.(nodes(s))
                push!(errs, maximum(abs, ud .- ex) / maximum(abs, ex))
                push!(hs, 2π / ne)
            end
            rate = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])
            @test rate > 1.8
        end
    end

    @testset "$(rpad("the gauged bracket is Poisson",76))" begin
        s = LagrangeSpace(2, 7)
        sys = BurgersSystem(s)
        u = uex.(nodes(s))
        @test isantisymmetric(sys.bracket, u)
        @test jacobi_residual(sys.bracket, u) < 1e-10
    end

    @testset "$(rpad("in the sqrt variables the bracket is constant and the Casimir linear",76))" begin
        s = LagrangeSpace(2, 7)
        u = uex.(nodes(s))
        ū = GeometricBrackets.to_sqrt_variables(u)
        @test GeometricBrackets.from_sqrt_variables(ū) ≈ u
        # C = 2 (∫φ_i) ⋅ ū exactly, which is why any symplectic method holds it at any step.
        # The 2 is the antiderivative of 1/g: G(u) = 2√u, and ū = √u.
        @test 2 * dot(basis_integrals(s), ū) ≈ hamiltonian(burgers_casimir(s), s, u)
    end

    @testset "$(rpad("the sqrt-variable bracket is K/4, not K -- the factor is pinned",76))" begin
        # The transformation and the factor move together: ū = √u goes with K/4, ū = 2√u with
        # K. Pairing 2√u with K/4 makes the flow four times too slow, and NO conservation
        # test can see it -- a constant rescaling of a Poisson field keeps every Casimir,
        # every energy and every convergence rate. Only a comparison against the pushforward
        # of the u-space field catches it, so that is what is checked here.
        s = LagrangeSpace(2, 13)
        u = uex.(nodes(s))
        Minv = inverse_mass_matrix(s)
        S = derivative_matrix(s)
        K = Minv * (S - transpose(S)) * Minv

        sys = BurgersSystem(s)
        udot = vectorfield(sys.flow, u)                      # u-space, reproduces 3 u u_x

        ū = GeometricBrackets.to_sqrt_variables(u)
        # d(√u)/dt = u̇ / (2√u), the pushforward of the u-space field
        ūdot = udot ./ (2 .* sqrt.(u))
        # ∂H/∂ū_i = 2 √u_i (M u)_i
        gū = 2 .* sqrt.(u) .* (mass_matrix(s) * u)

        @test (K * gū) ./ 4 ≈ ūdot
        @test !isapprox(K * gū, ūdot; rtol = 1e-3)          # the factor is not 1
    end

    @testset "$(rpad("semi-discrete conservation of H and C",76))" begin
        s = LagrangeSpace(2, 13)
        sys = BurgersSystem(s)
        u = uex.(nodes(s))
        f = vectorfield(sys.flow, u)
        @test abs(dot(gradient(sys.H, s, u), f)) < 1e-10
        @test abs(dot(gradient(sys.C, s, u), f)) < 1e-10
    end

    @testset "$(rpad("time integration",76))" begin
        s = LagrangeSpace(2, 13)
        sys = BurgersSystem(s)
        u0 = uex.(nodes(s))
        traj = integrate(sys, Integrator(sys.flow, ImplicitMidpoint(), 1e-3), u0, 200;
            stride = 20)
        @test drift(traj, :H) < 1e-10       # H is quadratic and the method symplectic
        @test drift(traj, :C) < 1e-6
    end
end
