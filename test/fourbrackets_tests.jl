using GeometricBrackets
using Test

@testset "$(rpad("Four-Bracket Tests",80))" begin

    # The manuscript's own test fields: band-limited, and generic -- no symmetry under
    # exchanging any two of them, which the sign tests of Lemma 3.1 below require.
    Au(g) = sample(g, (x, y) -> sin(x) + 0.6cos(2y) + 0.4sin(x + y) + 0.25cos(2x - 3y))
    Bu(g) = sample(g, (x, y) -> cos(y) + 0.5sin(2x) + 0.3cos(x - y) + 0.2sin(3x + y))
    Cu(g) = sample(g, (x, y) -> sin(x + 2y) + 0.4cos(3x) + 0.3sin(y))
    Du(g) = sample(g, (x, y) -> cos(2x - y) + 0.35sin(x))
    chi(g) = sample(g, (x, y) -> 2.0 + 0.5sin(x - y) + 0.3cos(2x + y))
    uu(g) = sample(g, (x, y) -> 3.0 + 0.4sin(x) + 0.3cos(y) + 0.2sin(x + 2y))

    @testset "$(rpad("each family satisfies exactly ONE antisymmetry condition",76))" begin
        # Lemma 3.1, and the sharpest claim in the manuscript: it is why there are two
        # families rather than one. The violations are asserted as firmly as the identities,
        # since a bracket antisymmetric in every slot would pass the vanishing half and mean
        # nothing at all.
        g = spectral_grid(16)
        a, b, c, d = Au(g), Bu(g), Cu(g), chi(g)

        gard = antisymmetry_residuals(gardner_4bracket, g, a, b, c, d)
        @test gard.slot_exchange < 1e-11
        @test gard.pair_exchange > 1e-3

        symm = antisymmetry_residuals(symmetric_4bracket, g, a, b, c, d)
        @test symm.slot_exchange > 1e-3
        @test symm.pair_exchange < 1e-11
    end

    @testset "$(rpad("the Gardner operators are antisymmetric and integrate by parts",76))" begin
        g = spectral_grid(16)
        a, b = Au(g), Bu(g)
        @test maximum(abs, gardner_x(g, a, b) .+ gardner_x(g, b, a)) < 1e-12
        @test maximum(abs, gardner_y(g, a, b) .+ gardner_y(g, b, a)) < 1e-12
        @test maximum(abs, gardner_x(g, a, a)) < 1e-12
        # flux plus a total derivative, eq. (4.14)
        @test maximum(abs, gardner_x(g, a, b) .-
                           (a .* ∂x(g, b) .- ∂x(g, a .* b) ./ 2)) < 1e-12
        # the curl identity, eq. (4.18), which is what makes the reduction possible
        @test maximum(abs,
            ∂y(g, gardner_x(g, a, b)) .- ∂x(g, gardner_y(g, a, b)) .+
            canonical_bracket(g, a, b)) < 1e-12
    end

    @testset "$(rpad("the symmetric operators ARE total derivatives",76))" begin
        # which is why the Section 6 bracket built from them vanishes identically, for any S
        g = spectral_grid(16)
        a, s = Au(g), chi(g)
        @test maximum(abs, symmetric_x(g, a, s) .- ∂x(g, a .* s)) < 1e-12
        @test maximum(abs, symmetric_y(g, a, s) .- ∂y(g, a .* s)) < 1e-12

        b = Bu(g)
        vanishes = integrate(g,
            symmetric_x(g, a, s) .* symmetric_y(g, b, s) .-
            symmetric_x(g, b, s) .* symmetric_y(g, a, s))
        scale = integrate(g, abs.(symmetric_x(g, a, s) .* symmetric_y(g, b, s)))
        @test abs(vanishes) / scale < 1e-12
    end

    @testset "$(rpad("a two-bracket IS its four-bracket with the entropy twice",76))" begin
        g = spectral_grid(16)
        a, b, s = Au(g), Bu(g), chi(g)
        @test gardner_2bracket(g, a, b, s) == gardner_4bracket(g, a, s, b, s)
        @test symmetric_2bracket(g, a, b, s) == symmetric_4bracket(g, a, s, b, s)
        @test gardner_2bracket_density(g, a, b, s) ==
              gardner_4bracket_density(g, a, s, b, s)
        @test symmetric_2bracket_density(g, a, b, s) ==
              symmetric_4bracket_density(g, a, s, b, s)
    end

    @testset "$(rpad("every bracket is the integral of its own density",76))" begin
        g = spectral_grid(16)
        a, b, c, d, s = Au(g), Bu(g), Cu(g), Du(g), chi(g)
        w = uu(g)
        @test gardner_4bracket(g, a, b, c, d) ≈
              integrate(g, gardner_4bracket_density(g, a, b, c, d))
        @test symmetric_4bracket(g, a, b, c, d) ≈
              integrate(g, symmetric_4bracket_density(g, a, b, c, d))
        @test weighted_2bracket(g, a, b, s, w) ≈
              integrate(g, weighted_2bracket_density(g, a, b, s, w))
        @test weighted_4bracket(g, a, b, c, d, w) ≈
              integrate(g, weighted_4bracket_density(g, a, b, c, d, w))
        # a unit weight is no weight at all
        one_ = fill(1.0, size(g))
        @test weighted_2bracket(g, a, b, s, one_) ≈ symmetric_2bracket(g, a, b, s)
        @test weighted_4bracket(g, a, b, c, d, one_) ≈ symmetric_4bracket(g, a, b, c, d)
    end

    @testset "$(rpad("both families reduce, and the factor of 1/2 is real",76))" begin
        g = spectral_grid(16)
        a, b, s = Au(g), Bu(g), chi(g)
        quadratic = integrate(g, s .^ 2 .* canonical_bracket(g, a, b))
        @test gardner_2bracket(g, a, b, s) ≈ quadratic / 2
        @test symmetric_2bracket(g, a, b, s) ≈ quadratic
        # so the two families reach int u[A_u,B_u] at different structure functions
        @test !isapprox(gardner_2bracket(g, a, b, s), symmetric_2bracket(g, a, b, s))
    end

    @testset "$(rpad("the Lie-Poisson bracket is reached EXACTLY, not asymptotically",76))" begin
        # at S_u = sqrt(2u) for the Gardner family and sqrt(u) for the symmetric one. u^{3/2}
        # is not band-limited, so this is a refinement claim.
        function residuals(f)
            r = Float64[]
            for N in (64, 128)
                g = finite_difference_grid(N)
                a, b, u = Au(g), Bu(g), uu(g)
                push!(r,
                    abs(f(g, a, b, u) - lie_poisson_2bracket(g, a, b, u)) /
                    abs(lie_poisson_2bracket(g, a, b, u)))
            end
            r
        end
        gard = residuals((g, a, b, u) -> gardner_2bracket(g, a, b, sqrt.(2 .* u)))
        symm = residuals((g, a, b, u) -> symmetric_2bracket(g, a, b, sqrt.(u)))
        @test gard[1] / gard[2] > 40
        @test symm[1] / symm[2] > 40
        @test gard[2] < 1e-9
        @test symm[2] < 1e-9
    end

    @testset "$(rpad("the Plücker relation holds POINTWISE, in two dimensions",76))" begin
        # Lemma 4.4: it is what collapses the twenty-four-term Jacobi obstruction to a single
        # condition on the structure function.
        g = spectral_grid(16)
        @test maximum(abs, plucker_residual(g, Au(g), Bu(g), Cu(g), uu(g))) < 1e-12
        # and it is a relation among the four arguments, not an accident of one of them
        @test maximum(abs, plucker_residual(g, Au(g), Bu(g), Cu(g), Du(g))) < 1e-12
    end

    @testset "$(rpad("Jacobi holds for EVERY structure function, on linear functionals",76))" begin
        # Theorem 4.5. Refinement, since log u and exp u are not band-limited.
        for (cfun, cprime) in ((u -> u, u -> 1.0),
            (u -> u^3, u -> 3u^2),
            (u -> (1 + log(u))^2 / 2, u -> (1 + log(u)) / u),
            (u -> exp(u), u -> exp(u)),
            (u -> sin(u), u -> cos(u)))
            r = map((64, 128)) do N
                g = finite_difference_grid(N)
                u = uu(g)
                jacobi_residual(g, Au(g), Bu(g), Cu(g), cfun.(u), cprime.(u))
            end
            @test r[1] / r[2] > 40
            @test r[2] < 1e-9
        end
    end

    @testset "$(rpad("an unnormalised residual reports its own scale",76))" begin
        g = finite_difference_grid(64)
        u = uu(g)
        res, scale = jacobi_residual(g, Au(g), Bu(g), Cu(g), u, fill(1.0, size(g));
            normalised = false)
        @test scale > 0
        @test res / scale ≈ jacobi_residual(g, Au(g), Bu(g), Cu(g), u, fill(1.0, size(g)))
    end
end
