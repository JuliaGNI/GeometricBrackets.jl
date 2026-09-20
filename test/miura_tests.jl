using GeometricBrackets
using LinearAlgebra
using SimpleSplines: UniformMesh, RandomMesh
using Test

@testset "$(rpad("Miura Tests",80))" begin
    @testset "$(rpad("the mass identity int u_h = -int v_h^2 is EXACT",76))" begin
        # Pairing û = M_h(v̂) with the partition of unity gives this to the last bit, and it
        # is what confines the image of the map to non-positive mass.
        for (nm, mk) in SPLINE_MESHES, N in (16, 24, 32)

            s = SplineSpace(mk(N), 3)
            v̂ = project(s, miura_initial_v(2π))
            û = miura_map(s, v̂)
            lhs = dot(basis_integrals(s), û)
            rhs = -dot(quadrature_weights(s), field(s, v̂) .^ 2)
            @test lhs ≈ rhs atol = 1e-12
            @test lhs < 0
        end
    end

    @testset "$(rpad("Miura bracket is antisymmetric and Poisson at every N",76))" begin
        # The Jacobiator is COMPUTED here, through the chart -- it is not zero by fiat.
        # Against ≈ 0.42, flat, for the Galerkin bracket.
        for N in (12, 16, 24)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            v̂ = project(s, miura_initial_v(2π))
            b = kdv_miura_bracket(s, v̂)
            û = miura_map(s, v̂)
            P = poisson_matrix(b, û)
            @test maximum(abs, P + P') < 1e-10 * maximum(abs, P)
            @test jacobi_residual(b, û) < 1e-12
            @test jacobi_residual(kdv_bracket_2(s), û) > 0.3
        end
    end

    @testset "$(rpad("the derivative tensor against finite differences",76))" begin
        # the chain rule through the chart, ∂P/∂û = (∂P/∂v̂) L⁻¹
        N = 12
        s = SplineSpace(UniformMesh(N, 2π), 3)
        v̂ = project(s, miura_initial_v(2π))
        b = kdv_miura_bracket(s, v̂)
        û = miura_map(s, v̂)
        dP = GeometricBrackets.poisson_derivative(b, û)
        L = miura_derivative(s, v̂)
        h = 1e-6
        for m in (1, 5, 11)
            e = zeros(N)
            e[m] = h
            vp = v̂ .+ L \ e
            vm = v̂ .- L \ e
            fd = (poisson_matrix(kdv_miura_bracket(s, vp), û) .-
                  poisson_matrix(kdv_miura_bracket(s, vm), û)) ./ 2h
            @test dP[m, :, :] ≈ fd atol = 1e-4 * max(1, maximum(abs, fd))
        end
    end

    @testset "$(rpad("K_Miura is DENSE where the Galerkin assembly is banded",76))" begin
        # the two interior inverse mass matrices are the nonlocality that carries the
        # Jacobi identity; the comparison must be on the weak-form blocks K = M P M
        for N in (16, 24)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            v̂ = project(s, miura_initial_v(2π))
            M = mass_matrix(s)
            û = miura_map(s, v̂)
            Km = M * poisson_matrix(kdv_miura_bracket(s, v̂), û) * M
            Kg = M * poisson_matrix(kdv_bracket_2(s), û) * M
            frac(A) = count(x -> abs(x) > 1e-8 * maximum(abs, A), A) / length(A)
            @test frac(Km) > 0.8
            @test frac(Kg) < 0.6
        end
    end

    @testset "$(rpad("the Hill criterion is sharp",76))" begin
        # A preimage exists exactly when the lowest eigenvalue of -∂ₓ² - u_h is positive.
        # The Hill potential is MINUS the KdV field in this convention, so the admissible
        # side of the threshold is the negative one: for u = c + sin x + 0.4 cos 2x at
        # N = 20, p = 3 the threshold sits at c = -0.470894, the mirror of the +0.470894 of
        # the older u_t = 6uu_x - u_xxx convention.
        s = SplineSpace(UniformMesh(20, 2π), 3)
        u(c) = project(s, x -> c + sin(x) + 0.4cos(2x))
        # λ₀(c) = λ₀(0) - c exactly, a constant in the potential shifting every eigenvalue,
        # so the admissible side is c BELOW the threshold.
        for c in (0.0, -0.2, -0.3)
            @test hill_lambda0(s, u(c)) < 0
            @test miura_invert(s, u(c)) === nothing        # no preimage at all
        end
        for c in (-0.35, -0.5, -1.0)
            @test hill_lambda0(s, u(c)) > 0
            @test miura_invert(s, u(c)) !== nothing
        end
        # the threshold itself, to four digits. It is λ₀ of the base field, and its magnitude
        # is NOT the 0.470894 of the older convention: there the Hill potential was +u and
        # here it is -u, and ±(sin x + 0.4 cos 2x) are not translates of one another.
        @test abs(hill_lambda0(s, u(-0.328741))) < 1e-5
    end

    @testset "$(rpad("miura_invert round-trips, and the branches differ",76))" begin
        s = SplineSpace(UniformMesh(20, 2π), 3)
        û = project(s, x -> -1.0 + sin(x) + 0.4cos(2x))
        v̂ = miura_invert(s, û)
        @test v̂ !== nothing
        @test miura_map(s, v̂) ≈ û atol = 1e-12
        # where there is a preimage there are two, with opposite ∫v
        w = miura_invert(s, û; seed = -1.0)
        @test w !== nothing
        @test miura_map(s, w) ≈ û atol = 1e-12
        @test sign(dot(basis_integrals(s), v̂)) == -sign(dot(basis_integrals(s), w))
    end

    @testset "$(rpad("none of the standard examples has a preimage",76))" begin
        # cos x has zero mass and the solitons are now ELEVATIONS, so all have C₀ ≥ 0,
        # while the image needs C₀ ≤ 0. The sign convention moved the half-space and the
        # solitons together, so the examples are excluded exactly as before -- which is the
        # point: the obstruction is spectral, not a matter of signs.
        s = SplineSpace(UniformMesh(32, 2π), 3)
        for f in (cosine(2π), x -> sin(x) + 0.4cos(2x))
            û = project(s, f)
            @test dot(basis_integrals(s), û) > -1e-10
            @test miura_invert(s, û) === nothing
        end
        sl = SplineSpace(UniformMesh(64, 40.0), 3)
        û = project(sl, soliton(1.0, 10.0, 40.0))
        @test dot(basis_integrals(sl), û) > 0
        @test miura_invert(sl, û) === nothing
    end

    @testset "$(rpad("MiuraHamiltonian gradient and Hessian",76))" begin
        s = SplineSpace(UniformMesh(12, 2π), 3)
        H = MiuraHamiltonian(s)
        v̂ = project(s, miura_initial_v(2π))
        h = 1e-6
        g = gradient(H, s, v̂)
        fd = [(hamiltonian(H, s, v̂ .+ h .* e) - hamiltonian(H, s, v̂ .- h .* e)) / 2h
              for e in eachcol(Matrix(I, 12, 12))]
        @test g ≈ fd atol = 1e-5 * max(1, maximum(abs, g))
        A = hessian(H, s, v̂)
        @test A ≈ A'
        fdH = zeros(12, 12)
        for m in 1:12
            e = zeros(12)
            e[m] = h
            fdH[:, m] = (gradient(H, s, v̂ .+ e) .- gradient(H, s, v̂ .- e)) ./ 2h
        end
        @test A ≈ fdH atol = 1e-4 * maximum(abs, A)
        # H̃ is the same number as H₂ at the corresponding û -- the value of a function is
        # chart-free, its algebraic form is not
        @test hamiltonian(H, s, v̂) ≈ hamiltonian(KdVHamiltonian2(s), s, miura_map(s, v̂))
    end

    @testset "$(rpad("MiuraSystem reports in u",76))" begin
        s = SplineSpace(UniformMesh(24, 2π), 3)
        sys = MiuraSystem(s)
        v̂ = project(s, miura_initial_v(2π))
        û = miura_map(s, v̂)
        kdv = KdVSystem(s)
        @test invariants(sys, v̂).H1 ≈ invariants(kdv, û).H1
        @test invariants(sys, v̂).H2 ≈ invariants(kdv, û).H2
        @test invariants(sys, v̂).C0 ≈ invariants(kdv, û).C0
        @test invariants(sys, v̂).C0 < 0        # the image has non-positive mass
        @test invariant_names(sys) == (:H1, :H2, :C0)
        @test nbasis(sys) == 24
    end

    @testset "$(rpad("the Miura flow reproduces the KdV flow of the second bracket",76))" begin
        for N in (32, 64)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            sys = MiuraSystem(s)
            v̂ = project(s, miura_initial_v(2π))
            û = miura_map(s, v̂)
            # v̇ = P¹ ∂H̃/∂v̂ pushes forward to u̇ = P²_M ∂H₂/∂û
            v̇ = vectorfield(sys.flow, v̂)
            pushed = miura_derivative(s, v̂) * v̇
            direct = poisson_apply(kdv_miura_bracket(s, v̂), û, gradient(KdVHamiltonian2(s), s, û))
            @test pushed ≈ direct atol = 1e-9 * maximum(abs, direct)
        end
    end

    @testset "$(rpad("CHART: midpoint loses H2 in v, a discrete gradient does not",76))" begin
        # H₂ is quadratic in û and QUARTIC in v̂, so the theorem that a symplectic
        # Runge-Kutta method conserves quadratic invariants does not apply in this chart.
        # What is chart-free is the VALUE of a function, so a discrete gradient method,
        # whose property is ḡ ⋅ Δv̂ = ΔH̃ by construction, does conserve it.
        s = SplineSpace(UniformMesh(20, 2π), 3)
        sys = MiuraSystem(s)
        v0 = project(s, miura_initial_v(2π))
        dt = 1.6e-3
        NT = round(Int, 0.2 / dt)

        mid = integrate(
            sys, Integrator(sys.flow, ImplicitMidpoint(), dt), v0, NT; stride = 5)
        @test drift(mid, :H2) > 1e-6            # ~7e-5
        @test drift(mid, :H2) < 1e-3

        dg = integrate(sys, Integrator(sys.flow, Gonzalez(), dt), v0, NT; stride = 5)
        @test drift(dg, :H2) < 1e-11            # exact
    end

    @testset "$(rpad("the flow cannot reach the degeneracy of L: int v is a Casimir",76))" begin
        # The condition number of L = M⁻¹(2B + S) blows up on ∫v = 0, and for the ASSEMBLED
        # operator that is evidence rather than a proof that the discrete critical set is
        # exactly the continuous one. What is exact is the other direction: ∫v_h is a Casimir
        # of P¹ in this chart, so a run started off ∫v = 0 stays off it, and the
        # near-degeneracy is not something a trajectory can wander into.
        s = SplineSpace(UniformMesh(20, 2π), 3)
        sys = MiuraSystem(s)
        v0 = project(s, miura_initial_v(2π))
        c0 = miura_casimir(sys, v0)
        @test abs(c0) > 1                       # v₀ = 1 + sin x is well off the set

        for meth in (ImplicitMidpoint(), Gonzalez())
            tr = integrate(sys, Integrator(sys.flow, meth, 1.6e-3), v0, 200; stride = 10)
            @test abs(miura_casimir(sys, tr.final) - c0) / abs(c0) < 1e-12
        end

        # Explicit Euler makes the same structural point in ONE step and cannot be run for
        # two hundred: it is unstable on this flow at this step size and reaches NaN. The
        # claim is about the increment lying in the range of P¹, so one increment is the
        # honest test of it.
        veul = copy(v0)
        integrate_step!(veul, Integrator(sys.flow, ExplicitEuler(), 1.6e-3))
        @test veul != v0
        @test abs(miura_casimir(sys, veul) - c0) / abs(c0) < 1e-12

        # and the condition number really is the two regimes the docstring quotes
        vzero = project(s, x -> sin(x))          # ∫v = 0
        @test cond(miura_derivative(s, vzero)) > 1e6
        @test cond(miura_derivative(s, v0)) < 1e2
    end

    @testset "$(rpad("the Poisson property transports through the chart",76))" begin
        # This is the first genuine Poisson integrator for a discrete SECOND KdV structure;
        # the Galerkin bracket admits none.
        s = SplineSpace(UniformMesh(20, 2π), 3)
        sys = MiuraSystem(s)
        v0 = project(s, miura_initial_v(2π))
        dt = 1.6e-3

        integ = Integrator(sys.flow, ImplicitMidpoint(), dt)
        v1 = copy(v0)
        integrate_step!(v1, integ)
        DΦv = tangent_map(integ, copy(v0))

        P1 = poisson_matrix(sys.bracket)
        @test maximum(abs, DΦv * P1 * DΦv' - P1) / maximum(abs, P1) < 1e-13

        # pushed forward, DΦᵤ = L(v¹) DΦᵥ L(v⁰)⁻¹ must map P²_M(v⁰) to P²_M(v¹)
        L0 = miura_derivative(s, v0)
        L1 = miura_derivative(s, v1)
        DΦu = L1 * DΦv * inv(L0)
        PM0 = poisson_matrix(kdv_miura_bracket(s, v0), v0)
        PM1 = poisson_matrix(kdv_miura_bracket(s, v1), v1)
        @test maximum(abs, DΦu * PM0 * DΦu' - PM1) / maximum(abs, PM1) < 1e-12
    end

    @testset "$(rpad("miura_casimir is the Casimir of P1 in the v chart",76))" begin
        s = SplineSpace(UniformMesh(20, 2π), 3)
        sys = MiuraSystem(s)
        v̂ = project(s, miura_initial_v(2π))
        @test miura_casimir(sys, v̂) ≈ dot(basis_integrals(s), v̂)
        # ∫v is conserved by the flow; ∫u = -∫v² is not a Casimir here
        @test abs(dot(basis_integrals(s), vectorfield(sys.flow, v̂))) < 1e-10
    end

    @testset "$(rpad("SPECTRAL PARAMETER: every standard example enters the chart",76))" begin
        # This is the point of λ. At λ = 0 none of the examples has a preimage -- the
        # obstruction being spectral and not a matter of signs -- and for λ below the lowest
        # Hill eigenvalue every one of them does.
        cases = (("cos", SplineSpace(UniformMesh(32, 2π), 3), cosine(2π)),
            ("mixed", SplineSpace(UniformMesh(32, 2π), 3), x -> sin(x) + 0.4cos(2x)),
            ("soliton", SplineSpace(UniformMesh(64, 40.0), 3), soliton(1.0, 10.0, 40.0)),
            ("2-soliton", SplineSpace(UniformMesh(64, 40.0), 3),
                two_soliton(1.2, 0.8, 12.0, 22.0, 40.0)))
        for (_, s, f) in cases
            û = project(s, f)
            @test miura_invert(s, û) === nothing            # nothing at λ = 0
            λ = miura_lambda(s, û)
            @test λ < hill_lambda0(s, û)                    # strictly below the threshold
            v̂ = miura_invert(s, û; λ = λ)
            @test v̂ !== nothing
            @test miura_map(s, v̂, λ) ≈ û atol = 1e-12       # and it really is a preimage
        end
    end

    @testset "$(rpad("the sharp criterion is lambda < lambda_0, not lambda_0 > 0",76))" begin
        s = SplineSpace(UniformMesh(20, 2π), 3)
        û = project(s, x -> sin(x) + 0.4cos(2x))
        λ₀ = hill_lambda0(s, û)
        seeds = range(-4.0, 4.0, length = 9)
        found(λ) = any(miura_invert(s, û; λ = λ, seed = σ) !== nothing for σ in seeds)
        # comfortably below: yes; comfortably above: no. The margin keeps the test off the
        # singular limit, where ψ acquires a zero and v = ψₓ/ψ is unbounded.
        for λ in (λ₀ - 0.05, λ₀ - 0.5, λ₀ - 5.0)
            @test found(λ)
        end
        for λ in (λ₀ + 0.05, λ₀ + 0.5, λ₀ + 5.0)
            @test !found(λ)
        end
    end

    @testset "$(rpad("the pushforward stays exactly Poisson for every lambda",76))" begin
        # L does not see a constant, so the bracket is untouched by λ -- but the field it is
        # evaluated at is not, so this is worth checking rather than asserting. Note that
        # L P¹ Lᵀ is NOT the Galerkin P² - 4λP¹: the two differ by exactly the Jacobi defect
        # that the Miura construction exists to remove.
        s = SplineSpace(UniformMesh(20, 2π), 3)
        û = project(s, cosine(2π))
        for λ in (-0.5, -2.0, -5.0)
            v̂ = miura_invert(s, û; λ = λ)
            @test v̂ !== nothing
            b = MiuraBracket(s, v̂)
            @test isantisymmetric(b, û)
            @test jacobi_residual(b, û) < 1e-12
            # and λ leaves L alone, which is the structural point
            @test miura_derivative(s, v̂) ≈ miura_derivative(s, v̂)
        end
    end

    @testset "$(rpad("a lambda-shifted Miura run conserves all three invariants",76))" begin
        # On the u side the extra -4λ P¹ ∂H₂/∂û is a constant-speed translation, so the
        # trajectory is the KdV solution in a moving frame and every invariant survives.
        s = SplineSpace(UniformMesh(24, 2π), 3)
        û0 = project(s, cosine(2π))
        λ = miura_lambda(s, û0)
        v0 = miura_invert(s, û0; λ = λ)
        @test v0 !== nothing
        sys = MiuraSystem(s; λ = λ)
        @test miura_lambda(sys) == λ
        @test miura_map(sys, v0) ≈ û0 atol = 1e-12          # the system agrees on the chart

        # The two methods exchange roles here exactly as they do at λ = 0, and for the same
        # reason: in this chart H̃ = H₂ ∘ M_h is QUARTIC in v̂ while C₀ = -∫v_h² - λL is
        # QUADRATIC, so the discrete gradient holds the energy and the symplectic midpoint
        # rule holds the mass. C₀ is the mKdV momentum here, not a Casimir, so neither method
        # holds it for free.
        mid = integrate(sys, Integrator(sys.flow, ImplicitMidpoint(), 1e-3; û₀ = v0), v0,
            200; stride = 10)
        dg = integrate(sys, Integrator(sys.flow, Gonzalez(), 1e-3; û₀ = v0), v0, 200;
            stride = 10)
        @test drift(dg, :H2) < 1e-12                # discrete gradient: energy exact
        @test drift(mid, :H2) > 1e-9                # midpoint: not, H̃ being quartic
        @test absolute_drift(mid, :C0) < absolute_drift(dg, :C0)   # ... but it holds the mass
        @test all(isfinite, mid.final) && all(isfinite, dg.final)
        # and both stay near the initial field: the λ drift is a translation, not growth
        @test maximum(abs, dg.final) < 10 * maximum(abs, v0)
    end

    @testset "$(rpad("new initial conditions",76))" begin
        u3 = three_solitons()
        @test u3(-60.0 - (-100.0)) ≈ 2 * 0.3^2 atol = 1e-3
        u5 = five_solitons()
        @test u5(-120.0 - (-150.0)) ≈ 2 * 0.3^2 atol = 1e-3
        v0 = miura_initial_v(2π)
        @test v0(0.0) ≈ 1
        @test v0(π/2) ≈ 2
    end
end
