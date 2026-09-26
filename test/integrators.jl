using GeometricBrackets
using LinearAlgebra
using SimpleSplines: UniformMesh
using Random
using Test
import SparseArrays

Random.seed!(0x5c1e9a3b)

@testset "$(rpad("Integrator Tests",80))" begin
    s = SplineSpace(UniformMesh(20, 2π), 3)
    sys = KdVSystem(s)
    u0 = project(s, cosine(2π))

    @testset "$(rpad("one step of every method runs and stays finite",76))" begin
        for meth in (ExplicitEuler(), RungeKutta4(), ImplicitMidpoint(),
            AverageVectorField(), Gonzalez(), GonzalezMass())
            for flow in (sys.flow1, sys.flow2)
                integ = Integrator(flow, meth, 1e-4)
                û = copy(u0)
                integrate_step!(û, integ)
                @test all(isfinite, û)
                @test û != u0
            end
        end
    end

    @testset "$(rpad("implicit midpoint reduces to the fixed point it defines",76))" begin
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
        û = copy(u0)
        integrate_step!(û, integ)
        r = û .- u0 .- 1e-3 .* vectorfield(sys.flow1, (u0 .+ û) ./ 2)
        @test maximum(abs, r) < 1e-11
    end

    @testset "$(rpad("the discrete gradient really is one",76))" begin
        # the defining property: ḡ ⋅ Δû = ΔH exactly, which is what makes energy
        # conservation follow from antisymmetry alone
        for m in (Gonzalez(), GonzalezMass())
            x = copy(u0)
            y = u0 .+ 0.05 .* randn(length(u0))
            ḡ = GeometricBrackets.discrete_gradient(m, sys.flow1, x, y)
            ΔH = hamiltonian(sys.flow1, y) - hamiltonian(sys.flow1, x)
            @test dot(ḡ, y .- x) ≈ ΔH atol = 1e-12 * max(1, abs(ΔH))
        end
    end

    @testset "$(rpad("the rank-one correction vanishes for a quadratic Hamiltonian",76))" begin
        # which is where the name "midpoint discrete gradient" comes from: for a quadratic
        # Hamiltonian the method IS the plain midpoint rule
        x = copy(u0)
        y = u0 .+ 0.05 .* randn(length(u0))
        for m in (Gonzalez(), GonzalezMass())
            ḡ = GeometricBrackets.discrete_gradient(m, sys.flow2, x, y)
            @test ḡ ≈ gradient(sys.flow2, (x .+ y) ./ 2) atol = 1e-10
        end
    end

    @testset "$(rpad("discrete gradient Jacobian against finite differences",76))" begin
        x = copy(u0)
        y = u0 .+ 0.05 .* randn(length(u0))
        for m in (Gonzalez(), GonzalezMass())
            J = GeometricBrackets.discrete_gradient_jacobian(m, sys.flow1, x, y)
            h = 1e-6
            for k in (1, 7, 15)
                e = zeros(length(y))
                e[k] = h
                fd = (GeometricBrackets.discrete_gradient(m, sys.flow1, x, y .+ e) .-
                      GeometricBrackets.discrete_gradient(m, sys.flow1, x, y .- e)) ./ 2h
                @test J[:, k] ≈ fd atol = 1e-4 * max(1, maximum(abs, fd))
            end
        end
    end

    @testset "$(rpad("tangent map against finite differences",76))" begin
        for meth in (ImplicitMidpoint(), AverageVectorField(), Gonzalez(), ExplicitEuler())
            integ = Integrator(sys.flow1, meth, 1e-3)
            DΦ = tangent_map(integ, copy(u0))
            h = 1e-6
            for k in (1, 9, 17)
                e = zeros(length(u0))
                e[k] = h
                yp = u0 .+ e
                ym = u0 .- e
                integrate_step!(yp, integ)
                integrate_step!(ym, integ)
                @test DΦ[:, k] ≈ (yp .- ym) ./ 2h atol = 1e-5
            end
        end
    end

    @testset "$(rpad("implicit midpoint IS a Poisson map, AVF is not",76))" begin
        # P¹ is constant, so implicit midpoint is the Cayley transform of P¹ times a
        # symmetric matrix, which preserves P¹ exactly. Ge-Marsden then says no method can
        # be both this and exactly energy-preserving.
        mid = Integrator(sys.flow1, ImplicitMidpoint(), 2e-3)
        avf = Integrator(sys.flow1, AverageVectorField(), 2e-3)
        dgr = Integrator(sys.flow1, Gonzalez(), 2e-3)
        eul = Integrator(sys.flow1, ExplicitEuler(), 2e-3)
        @test poisson_defect(mid, copy(u0)) < 1e-13
        @test poisson_defect(avf, copy(u0)) > 1e-7
        @test poisson_defect(dgr, copy(u0)) > 1e-9
        @test poisson_defect(eul, copy(u0)) > 1e-2
    end

    @testset "$(rpad("stability limit and explicit stability",76))" begin
        dtmax = GeometricBrackets.stability_limit(sys.flow1, copy(u0))
        @test dtmax > 0
        # stable at 40% of the limit, unstable at 160%
        for (factor, stable) in ((0.4, true), (1.6, false))
            integ = Integrator(sys.flow1, RungeKutta4(), factor * dtmax)
            û = copy(u0)
            for _ in 1:400
                integrate_step!(û, integ)
                all(isfinite, û) || break
            end
            grew = !all(isfinite, û) || maximum(abs, û) > 100 * maximum(abs, u0)
            @test grew == !stable
        end
    end

    @testset "$(rpad("the spectral radius reproduces the values quoted in the notes",76))" begin
        # Section 5 of discrete-kdv-brackets-notes.tex quotes rho = 115, 274, 907, 2170 at
        # N = 12, 16, 24, 32 for p = 3 on the first flow, and kdvsim.py checks the same four
        # figures. This is the one place where the step size of every explicit run in the
        # manuscript is pinned, so a change in the assembly should fail here rather than
        # show up as a silent disagreement with the notes.
        ρ_ref = (12 => 115.0, 16 => 274.0, 24 => 907.0, 32 => 2170.0)
        ρ = Float64[]
        for (N, ref) in ρ_ref
            sN = SplineSpace(N, 3; L = 2π)
            sysN = KdVSystem(sN)
            r = 2 * sqrt(2) /
                GeometricBrackets.stability_limit(sysN.flow1, project(sN, sin))
            push!(ρ, r)
            @test isapprox(r, ref; rtol = 5e-3)
        end
        # and it grows as h^-3
        for (i, (a, b)) in enumerate(((12, 16), (16, 24), (24, 32)))
            @test isapprox(ρ[i + 1] / ρ[i], (b / a)^3; rtol = 0.03)
        end
    end

    @testset "$(rpad("projection method holds several invariants at once",76))" begin
        # projecting onto both Hamiltonians alone LOSES the Casimir, because the correction
        # direction is not mass-neutral; a third multiplier along g restores all three
        both = ProjectionMethod(ImplicitMidpoint(), (sys.H1, sys.H2))
        three = ProjectionMethod(ImplicitMidpoint(), (sys.H1, sys.H2, sys.C0))

        t2 = integrate(sys, Integrator(sys.flow1, both, 2e-3), u0, 100; stride = 10)
        @test drift(t2, :H1) < 1e-12
        @test drift(t2, :H2) < 1e-12

        t3 = integrate(sys, Integrator(sys.flow1, three, 2e-3), u0, 100; stride = 10)
        @test drift(t3, :H1) < 1e-12
        @test drift(t3, :H2) < 1e-12
        @test absolute_drift(t3, :C0) < 1e-12
    end

    @testset "$(rpad("LapackLU and LU agree, so the substitution is a performance one",76))" begin
        # Both solvers come from SimpleSolvers; LapackLU replaces only the factorisation
        # inside its Newton loop, so the two must produce the same step -- it is a
        # performance substitution, not a different method.
        import SimpleSolvers
        for meth in (ImplicitMidpoint(), AverageVectorField(), Gonzalez())
            a = copy(u0)
            b = copy(u0)
            integrate_step!(a, Integrator(sys.flow1, meth, 1e-3;
                linear_solver_method = LapackLU()))
            integrate_step!(b, Integrator(sys.flow1, meth, 1e-3;
                linear_solver_method = SimpleSolvers.LU()))
            @test a ≈ b atol = 1e-11
        end
        # and over a run
        ta = integrate(sys,
            Integrator(sys.flow1, ImplicitMidpoint(), 1e-3;
                linear_solver_method = LapackLU()),
            u0,
            100;
            stride = 10)
        tb = integrate(sys,
            Integrator(sys.flow1, ImplicitMidpoint(), 1e-3;
                linear_solver_method = SimpleSolvers.LU()),
            u0,
            100;
            stride = 10)
        @test ta.final ≈ tb.final atol = 1e-10
        @test drift(ta, :H1) ≈ drift(tb, :H1) rtol = 1e-6
    end

    @testset "$(rpad("the mixed two-field formulation reproduces the dense one",76))" begin
        # The mixed formulation solves 2N equations in (y, v) instead of N in y, and never
        # forms Minv*K*Minv. Its residual is the dense one premultiplied by the mass matrix,
        # so it describes the *same* step: the two must agree to solver tolerance, and the
        # substitution is a performance one exactly as LapackLU-for-LU was.
        for flow in (sys.flow2,)          # AffineBracket only; see `kernel_operator`
            a = copy(u0)
            b = copy(u0)
            integrate_step!(a, Integrator(flow, ImplicitMidpoint(), 1e-3; û₀ = u0))
            integrate_step!(b, Integrator(flow, ImplicitMidpoint(), 1e-3; û₀ = u0,
                formulation = :mixed))
            @test a ≈ b atol = 1e-11
            @test a != u0
        end

        # and over a run, including the conserved quantities, which is the real test: an
        # equivalent step that drifted differently would mean the formulations are not
        # actually the same method.
        ta = integrate(sys, Integrator(sys.flow2, ImplicitMidpoint(), 1e-3; û₀ = u0),
            u0, 100; stride = 10)
        tb = integrate(sys,
            Integrator(sys.flow2, ImplicitMidpoint(), 1e-3; û₀ = u0,
                formulation = :mixed),
            u0,
            100; stride = 10)
        @test ta.final ≈ tb.final atol = 1e-10
        # H2 is this flow's Hamiltonian and C0 its Casimir, and both are held to round-off.
        # H1 is *not* conserved on the second bracket -- it drifts at 5.6e-5 over these 100
        # steps -- so the test is that the two formulations drift the *same*, which is the
        # statement that they are the same method. Asserting an absolute bound on H1 here
        # would be asserting something false about implicit midpoint on flow2.
        @test drift(tb, :H2) < 1e-12
        @test drift(ta, :H2) < 1e-12
        @test absolute_drift(tb, :C0) < 1e-12
        @test drift(ta, :H1) ≈ drift(tb, :H1) rtol = 1e-6
    end

    @testset "$(rpad("the mixed formulation keeps its Jacobian sparse",76))" begin
        import SimpleSolvers
        import SparseArrays
        integ = Integrator(sys.flow2, ImplicitMidpoint(), 1e-3; û₀ = u0,
            formulation = :mixed)
        N = nbasis(s)
        J = SimpleSolvers.jacobianmatrix(SimpleSolvers.cache(integ.solver))
        @test J isa SparseArrays.SparseMatrixCSC
        @test size(J) == (2N, 2N)
        # the pattern is fixed for the whole run, which is what lets the solver reuse one
        # symbolic factorization -- and `mixed_jacobian!` errors if it ever grows
        nz = SparseArrays.nnz(J)
        u = copy(u0)
        for _ in 1:5
            integrate_step!(u, integ)
            @test SparseArrays.nnz(J) == nz
        end
        # and it is genuinely sparse: the stored count grows like N, not N². At the N = 20 of
        # this testset a banded 2N × 2N matrix is still a third full, so the meaningful
        # assertion is the scaling rather than a fraction at one size.
        s2 = SplineSpace(UniformMesh(4 * N, 2π), degree(s))
        sys2 = KdVSystem(s2)
        integ2 = Integrator(sys2.flow2, ImplicitMidpoint(), 1e-3;
            û₀ = project(s2, cosine(2π)), formulation = :mixed)
        nz2 = SparseArrays.nnz(SimpleSolvers.jacobianmatrix(SimpleSolvers.cache(integ2.solver)))
        @test nz2 / nz ≈ 4 rtol = 0.05        # O(N), not O(N²) -- which would be 16
        # SparspakLU, not the UmfpackLU SimpleSolvers would pick for a sparse Float64
        # Jacobian: UMFPACK mis-handles this block structure from N = 768 up. See the
        # comment in the Integrator constructor.
        @test SimpleSolvers.method(SimpleSolvers.linearsolver(integ.solver)) isa
              SimpleSolvers.SparspakLU
        # and an explicit method still wins over that choice
        integ_u = Integrator(sys.flow2, ImplicitMidpoint(), 1e-3; û₀ = u0,
            formulation = :mixed,
            linear_solver_method = SimpleSolvers.UmfpackLU())
        @test SimpleSolvers.method(SimpleSolvers.linearsolver(integ_u.solver)) isa
              SimpleSolvers.UmfpackLU
    end

    @testset "$(rpad("the mixed formulation refuses what it cannot do",76))" begin
        # Only ImplicitMidpoint is implemented: AverageVectorField needs one auxiliary per
        # quadrature node and the discrete-gradient methods carry a rank-one term.
        for meth in (AverageVectorField(), Gonzalez(), GonzalezMass())
            @test_throws ArgumentError Integrator(sys.flow2, meth, 1e-3;
                formulation = :mixed)
        end
        # and only an AffineBracket: the others store the sandwiched Minv*K*Minv, so the
        # weak-form block cannot be recovered
        @test_throws ArgumentError Integrator(sys.flow1, ImplicitMidpoint(), 1e-3;
            formulation = :mixed)
        @test_throws ArgumentError Integrator(sys.flow2, ImplicitMidpoint(), 1e-3;
            formulation = :nonsense)
    end

    @testset "$(rpad("an unconverged step is redone with a line search",76))" begin
        # The fast path takes a full Newton step. Where that fails -- implicit midpoint on
        # the second flow does, on about 1 % of the steps of the large-amplitude Miura
        # initial condition -- the step is redone from uⁿ with a line search rather than
        # accepted as it stands.
        sm = SplineSpace(64, 3; L = 2π)
        sysm = KdVSystem(sm)
        w0 = miura_map(sm, project(sm, miura_initial_v(2π)))

        integ = Integrator(sysm.flow2, ImplicitMidpoint(), 1e-3)
        @test integ.fallback !== integ.solver
        @test integ.fallbackstate !== integ.state

        # 400 steps run without a single warning reaching the user
        w = copy(w0)
        @test_nowarn for _ in 1:400
            integrate_step!(w, integ)
        end
        @test all(isfinite, w)

        # and the trajectory matches the one a line search would have produced throughout
        back = Integrator(sysm.flow2, ImplicitMidpoint(), 1e-3;
            linesearch = SimpleSolvers.Backtracking(Float64))
        wb = copy(w0)
        for _ in 1:400
            integrate_step!(wb, back)
        end
        @test maximum(abs, w .- wb) < 1e-8 * maximum(abs, wb)
    end

    @testset "$(rpad("solver is built once and reused across steps",76))" begin
        integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
        solver = integ.solver
        û = copy(u0)
        for _ in 1:5
            integrate_step!(û, integ)
        end
        @test integ.solver === solver
    end
end
