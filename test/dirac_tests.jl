using GeometricBrackets
using LinearAlgebra
using Random
using Test

# A point on the constraint surface: u_a = 0 for every constrained direction a.
function surface_point(rng, N, keep)
    u = zeros(Rational{BigInt}, N)
    for i in keep
        u[i] = rand(rng, -6:6) // rand(rng, 1:4)
    end
    u
end

@testset "$(rpad("Dirac Reduction Tests",80))" begin
    @testset "$(rpad("the structural identities of the Dirac tensor hold EXACTLY",76))" begin
        rng = MersenneTwister(0xd15ac)
        for tr in (witt_truncation(1), witt_truncation(2))
            C, keep, con = tr.C, tr.keep, tr.con
            N = size(C, 1)
            u = surface_point(rng, N, keep)
            J = lie_poisson_matrix(C, u)

            @test length(maximal_second_class(J, con)) == length(con)   # fully second class

            R = dirac_R(J, con)
            Js = dirac_tensor(J, con)
            @test R * R == R                                    # idempotent
            @test all(iszero, R[con, :])                        # V₂ rows vanish identically
            @test exact_rank(R) == length(keep)
            @test Js == R * J * transpose(R)
            @test Js == -transpose(Js)
            @test Js[keep, keep] == schur_complement(J, keep, con)
            # the constraints are Casimirs of J*: that is the point of the construction
            @test all(iszero, Js[:, con])
            @test all(iszero, Js[con, :])

            P = dirac_projector(J, keep, con)
            @test size(P) == (length(keep), N)
            @test P == R[keep, :]
        end
    end

    @testset "$(rpad("[Jhat,Jhat] = P⊗P⊗P [J,J]: the Jacobiator transports TENSORIALLY",76))" begin
        # The central claim of the Dirac companion note, and the reason Dirac reduction
        # preserves the Jacobi identity but can never create it.
        rng = MersenneTwister(0x7ac0b1)
        for tr in (witt_truncation(1), witt_truncation(2))
            C, keep, con = tr.C, tr.keep, tr.con
            for _ in 1:3
                u = surface_point(rng, size(C, 1), keep)
                J = lie_poisson_matrix(C, u)
                length(maximal_second_class(J, con)) == length(con) || continue

                Ĵ, dĴ = reduced_bracket(C, u, keep, con)
                @test Ĵ == -transpose(Ĵ)
                lhs = jacobiator(Ĵ, dĴ)
                rhs = project_jacobiator(jacobiator(J, lie_poisson_derivative(C)),
                    dirac_projector(J, keep, con))
                @test lhs == rhs                    # exactly, over ℚ
            end
        end
    end

    @testset "$(rpad("reduction does NOT repair the truncated Witt algebra at K = 2",76))" begin
        rng = MersenneTwister(0x21ac)
        tr = witt_truncation(2)
        C, keep, con = tr.C, tr.keep, tr.con
        u = surface_point(rng, size(C, 1), keep)
        J = lie_poisson_matrix(C, u)
        Ĵ, dĴ = reduced_bracket(C, u, keep, con)
        @test jacobi_residual(J, lie_poisson_derivative(C)) != 0
        @test jacobi_residual(Ĵ, dĴ) != 0
    end

    @testset "$(rpad("second-class subsets grow in PAIRS: an odd block is always singular",76))" begin
        # The diagonal of an antisymmetric matrix vanishes, so a single constrained direction
        # is never usable. This parity is one of the two obstructions raised before any
        # computation in the Dirac note.
        rng = MersenneTwister(0x0dd)
        for _ in 1:10
            n = 7
            A = zeros(Rational{BigInt}, n, n)
            for i in 1:n, j in (i + 1):n

                v = rand(rng, -6:6) // rand(rng, 1:4)
                A[i, j], A[j, i] = v, -v
            end
            for con in ([1, 2, 3], [1, 2, 3, 4, 5], 1:7)
                S = maximal_second_class(A, con)
                @test iseven(length(S))
                @test S ⊆ collect(con)
                @test issorted(S)
                isempty(S) || @test det(A[S, S]) != 0
            end
        end
    end

    @testset "$(rpad("restrict_c is the naive truncation, and differs unless V2 is an ideal",76))" begin
        tr = witt_truncation(1)
        C, keep, con = tr.C, tr.keep, tr.con
        Cn = restrict_c(C, keep)
        @test size(Cn) == (length(keep), length(keep), length(keep))
        @test Cn == C[keep, keep, keep]
        @test !is_ideal(C, con, keep)
    end

    @testset "$(rpad("dschur refuses to differentiate along a constrained direction",76))" begin
        rng = MersenneTwister(0xd5c)
        tr = witt_truncation(1)
        C, keep, con = tr.C, tr.keep, tr.con
        u = surface_point(rng, size(C, 1), keep)
        @test_throws ArgumentError dschur(C, u, keep, con, first(con))
        @test size(dschur(C, u, keep, con, first(keep))) == (length(keep), length(keep))
    end

    @testset "$(rpad("the hierarchical FEM basis closes on V1 exactly, unlike a C0 one",76))" begin
        for (p, ne) in ((1, 3), (1, 4), (2, 2), (2, 3))
            r = assemble_dg_hierarchical(p, ne)
            @test size(r.C, 1) == 2p * ne
            @test length(r.keep) == p * ne
            @test length(r.con) == p * ne
            @test is_antisymmetric_c(r.C)
            @test closes_on(r.C, r.keep, r.con)
            @test leak(r.C, r.keep, r.con) == 0        # exact, because V1·V1' ⊆ P_{2p-1}
            @test r.M == transpose(r.M)
        end
    end

    @testset "$(rpad("REGRESSION: exact agreement with the Python prototypes, digit for digit",76))" begin
        # Values produced by `diractools.py` and `femtools.py` at a fixed rational point,
        # with no RNG anywhere. They are exact rationals, so this is not a tolerance test:
        # any drift in the Dirac machinery or the hierarchical assembly changes them.
        # `dg23_M_00 = 4//15` in particular pins the exact shifted-Legendre and Lagrange
        # assembly of src/hierarchical.jl against the SymPy version it replaces.
        Q = Rational{BigInt}
        function upt(N, keep)
            u = zeros(Q, N)
            for (t, i) in enumerate(keep)
                u[i] = Q(t) // Q(((t - 1) % 3) + 2)
            end
            u
        end
        raw(J, dJ) = first(jacobi_residual(J, dJ; normalised = false))

        tr = witt_truncation(2)
        C, keep, con = tr.C, tr.keep, tr.con
        u = upt(size(C, 1), keep)
        J, dJ = lie_poisson_matrix(C, u), lie_poisson_derivative(C)
        Ĵ, dĴ = reduced_bracket(C, u, keep, con)
        @test det(J[con, con]) == 13225 // 9
        @test raw(J, dJ) == 54
        @test raw(Ĵ, dĴ) == 33145 // 1587
        @test maximum(abs, Ĵ) == 165 // 46
        @test Ĵ[1, 2] == 0

        r = assemble_dg_hierarchical(2, 3)
        C3, keep3, con3, M3 = r.C, r.keep, r.con, r.M
        u3 = upt(size(C3, 1), keep3)
        J3 = lie_poisson_matrix(C3, u3)
        Ĵ3, dĴ3 = reduced_bracket(C3, u3, keep3, con3)
        @test M3[1, 1] == 4 // 15
        @test first(structure_constant_residual(C3; normalised = false)) == 10368 // 49
        @test det(J3[con3, con3]) == 10490880625 // 50176
        @test raw(Ĵ3, dĴ3) == 16134943910670 // 117497863
        @test maximum(abs, Ĵ3) == 520412685 // 229432
    end

    @testset "$(rpad("shifted Legendre is orthogonal and Lagrange is nodal, both exactly",76))" begin
        @test shifted_legendre(3) == [[1], [-1, 2], [1, -6, 6], [-1, 12, -30, 20]]
        M, T3 = dg_local_algebra(2)
        @test M == Diagonal([1 // (2a - 1) for a in 1:4])
        # T is antisymmetric in its last two slots by construction
        @test all(T3[m, a, b] == -T3[m, b, a] for m in 1:4, a in 1:4, b in 1:4)

        for p in (1, 2, 3)
            φ = lagrange_coefficients(p)
            ev(c, x) = sum(c[k] * x^(k - 1) for k in eachindex(c))
            @test all(ev(φ[a], (b - 1) // p) == (a == b) for a in 1:(p + 1), b in 1:(p + 1))
            @test all(sum(ev(φ[a], x) for a in 1:(p + 1)) == 1 for x in (0//1, 1//3, 1//1))
        end
        @test_throws ArgumentError lagrange_coefficients(0)
    end
end
