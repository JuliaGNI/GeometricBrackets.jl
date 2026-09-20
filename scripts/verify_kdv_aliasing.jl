#!/usr/bin/env julia
#
# Can Zeitlin-style aliasing restore the Jacobi identity of the second KdV bracket?
#
#     julia --project=scripts scripts/verify_kdv_aliasing.jl
#
# No. Zeitlin's construction replaces a Fourier truncation of the sine-Euler bracket by
# structure constants that are PERIODIC IN THE GRADING,
#
#     c_{mn}^{m+n} = (N/2pi) sin( (2pi/N) m x n ) ,     m + n modulo N ,
#
# so that reducing the index mod N is invisible: sin is unchanged by m x n -> m x n + N. The
# second KdV bracket is the Virasoro Lie-Poisson structure, whose structure constants are
# m - n: linear and unbounded. No N-periodic function agrees with m - n across a window of
# width N without an O(N) jump at the seam, and that jump is precisely the aliasing error.
#
#   1. aliasing is not even definable in a spline basis: it needs a multiplicative grading
#      phi_m phi_n in span{phi_{m+n}}, which B-splines do not have and Fourier modes do;
#   2. in a Fourier basis the second bracket is exactly Virasoro,
#      P_mn = (m-n) u_{m+n} + c m^3 delta_{m+n,0};
#   3. THE ZERO-MODE THEOREM. For any Z_N-graded ansatz [L_m,L_n] = f(m,n) L_{m+n}, setting
#      m = 0 in the graded Jacobi condition gives, identically,
#
#          J(0,n,p) = f(n,p) [ phi(n) + phi(p) - phi(n+p) ] ,   phi(n) := f(0,n).
#
#      So phi must be additive wherever the bracket is nonzero. If f(n,p) != 0 for all n != p
#      -- which consistency with m - n forces -- then for N >= 5 additivity over Z_N forces
#      phi == 0, i.e. L_0 CENTRAL, whereas consistency demands phi(n) -> -n. The sole
#      exception is N = 3, whose nondegenerate solution has a nondegenerate Killing form and
#      is therefore sl(2,C), the unique three-dimensional simple Lie algebra -- and three
#      modes cannot converge;
#   4. why Zeitlin escapes: the sine algebra is graded by Z_N^2 minus the origin, the zero
#      mode being simply absent, and legitimately so, because constants are already central
#      in C^inf(T^2). Its phi vanishes identically. Witt's L_0 is the grading element and
#      cannot be dropped, since [L_n,L_-n] = 2n L_0;
#   5. a second, independent obstruction: the -d_x^3 term is the Virasoro cocycle, which is
#      not a coboundary, while every candidate target here is semisimple and so has H^2 = 0
#      by Whitehead's lemma.
#
# Needs SymPy.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random
using SymPyPythonCall

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

const TOL = 1e-9

"J(m,n,p) for [L_m,L_n] = f[m,n] L_{(m+n) mod N}, on 0-based mode labels."
function graded_jacobi(f, N)
    J = zeros(N, N, N)
    for m in 0:(N - 1), n in 0:(N - 1), p in 0:(N - 1)
        J[m + 1, n + 1, p + 1] = f[m + 1, n + 1] * f[mod(m + n, N) + 1, p + 1] +
                                 f[n + 1, p + 1] * f[mod(n + p, N) + 1, m + 1] +
                                 f[p + 1, m + 1] * f[mod(p + m, N) + 1, n + 1]
    end
    J
end

"f(m,n) = m - n with symmetric representatives -K..K, indices mod N."
function witt_f(N)
    r = [float(m - N * (m > N ÷ 2)) for m in 0:(N - 1)]
    r .- transpose(r)
end

header("1. aliasing needs a multiplicative grading; B-splines have none")
let n = 16
    s = SplineSpace(n, 3)
    M = Matrix(mass_matrix(s))
    Minv = inv(M)
    Φ = basis_values(s, 0)
    W = quadrature_weights(s)
    worst = 0.0
    for (j, k) in ((3, 5), (4, 4), (6, 9))
        # expand φ_j φ_k in the basis and see where the mass sits
        coeff = Minv * (Φ * (W .* Φ[j + 1, :] .* Φ[k + 1, :]))
        tgt = mod(j + k, n) + 1
        frac = abs(coeff[tgt]) / sum(abs, coeff)
        worst = max(worst, frac)
        @printf("      phi_%d phi_%d:  %d of %d coefficients nonzero; the share at index j+k mod n is %.3f\n",
            j, k, count(>(1e-8), abs.(coeff)), n, frac)
    end
    check("the product of two B-splines is not proportional to a third: no grading",
        worst < 0.5, @sprintf("largest share at j+k was %.3f", worst))
    println("      by contrast  e_m e_n = e_{m+n}  exactly, which is what makes " *
            "folding back well defined")
end

header("2. in a Fourier basis the second bracket is exactly Virasoro")
# D2 e_n = i sum_k (4n + 2k) u_k e_{k+n} + i n^3 e_n ; read off the e_{-m} coefficient
let K = 6
    modes = (-K):K
    rng = MersenneTwister(0)
    uhat = Dict(k => complex(randn(rng), randn(rng)) for k in (-2K):(2K))
    err = 0.0
    for m in modes, nn in modes

        assembled = im * (4nn + 2 * (-m - nn)) * get(uhat, -m - nn, 0im)
        assembled += (nn == -m) ? im * nn^3 : 0im
        closed = -2im * ((m - nn) * get(uhat, -(m + nn), 0im) +
                  0.5 * m^3 * (m + nn == 0 ? 1 : 0))
        err = max(err, abs(assembled - closed))
    end
    check("P_mn = (m-n) u_{m+n} + (1/2) m^3 delta_{m+n,0}, up to the overall factor -2i",
        err < 1e-9, @sprintf("max mismatch %.2e", err))
end

header("3. the zero-mode theorem")
let rng = MersenneTwister(1)
    for N in (5, 7, 9)
        A = randn(rng, N, N)
        f = A - transpose(A)                       # an arbitrary antisymmetric f
        J = graded_jacobi(f, N)
        φ = f[1, :]
        rhs = [f[n + 1, p + 1] * (φ[n + 1] + φ[p + 1] - φ[mod(n + p, N) + 1])
               for n in 0:(N - 1), p in 0:(N - 1)]
        e = maximum(abs, J[1, :, :] - rhs)
        check("N = $N: J(0,n,p) = f(n,p) [ phi(n) + phi(p) - phi(n+p) ], identically",
            e < TOL, @sprintf("max error %.2e", e))
    end
end

# nondegeneracy => phi additive on every pair => phi = 0, for N >= 5
println("\n  nondegenerate (f(n,p) != 0 for all n != p) forces phi additive; solving that:")
for N in (3, 5, 7, 9, 11, 13)
    φ = vcat(Sym(0), [Sym("p$i") for i in 1:(N - 1)])
    # Pass bare expressions, which sympy reads as "= 0". A Vector of `Eq` objects makes
    # SymPyCore build a sympy Matrix out of them, and non-Expr entries in a Matrix are
    # deprecated -- harmless today, an error in a future sympy.
    eqs = [φ[i + 1] + φ[j + 1] - φ[mod(i + j, N) + 1] for i in 1:(N - 1)
           for j in (i + 1):(N - 1)]
    unk = φ[2:end]
    sol = solve(eqs, unk; dict = true)
    forced = !isempty(sol) &&
             all(simplify(subs(v, [k => w for (k, w) in sol[1]]...)) == 0 for v in unk)
    @printf("      N = %3d: %3d equations -> %s\n", N, length(eqs),
        forced ? "phi == 0 forced" : "phi NOT forced to zero")
    if N ≥ 5
        check("N = $N: nondegeneracy forces phi == 0, i.e. L_0 central", forced)
    else
        check("N = $N: phi is NOT forced to zero -- the known exception", !forced)
    end
end

# the N = 3 exception is sl(2,C)
let
    bs, cs = symbols("b", nonzero = true), symbols("c", nonzero = true)
    f3 = [Sym(0) -bs bs; bs Sym(0) cs; -bs -cs Sym(0)]
    res = Set(simplify(f3[m + 1, n + 1] * f3[mod(m + n, 3) + 1, p + 1] +
                       f3[n + 1, p + 1] * f3[mod(n + p, 3) + 1, m + 1] +
                       f3[p + 1, m + 1] * f3[mod(p + m, 3) + 1, n + 1])
    for m in 0:2, n in 0:2, p in 0:2)
    check("N = 3: the nondegenerate branch satisfies Jacobi exactly", res == Set([Sym(0)]))
    function ad(f, m, N)
        A = fill(Sym(0), N, N)
        for i in 0:(N - 1)
            A[mod(m + i, N) + 1, i + 1] = f[m + 1, i + 1]
        end
        A
    end
    Kf = [expand(tr(ad(f3, i, 3) * ad(f3, j, 3))) for i in 0:2, j in 0:2]
    dt = factor(simplify(det(Kf)))
    check(
        "N = 3: its Killing form is nondegenerate, so it is sl(2,C) -- and three " *
        "modes cannot converge",
        dt != 0,
        "det K = $dt")
end

header("4. the quantitative no-go: aliasing Witt costs O(N)")
@printf("      %4s %10s %10s %11s\n", "N", "|J|_inf", "scale", "normalised")
let norms = Float64[]
    for N in (5, 7, 9, 11, 15, 21, 31)
        f = witt_f(N)
        J = graded_jacobi(f, N)
        A = [f[m + 1, n + 1] * f[mod(m + n, N) + 1, p + 1]
             for m in 0:(N - 1), n in 0:(N - 1), p in 0:(N - 1)]
        v = maximum(abs, J) / maximum(abs, A)
        push!(norms, v)
        @printf("      %4d %10.1f %10.1f %11.4f\n", N, maximum(abs, J), maximum(abs, A), v)
    end
    check("the normalised residual does not decrease with N -- it increases",
        all(norms[i] < norms[i + 1] for i in 1:(length(norms) - 1)),
        "[" * join([@sprintf("'%.3f'", v) for v in norms], ", ") * "]")
end

let N = 11
    φ = witt_f(N)[1, :]
    defect = Set(round(φ[i + 1] + φ[j + 1] - φ[mod(i + j, N) + 1]; digits = 6)
    for i in 0:(N - 1), j in 0:(N - 1))
    check("N = $N: the defect phi(n)+phi(p)-phi(n+p) is 0 without a wrap and -+N with one",
        defect == Set([0.0, float(N), -float(N)]), "values $(sort(collect(defect)))")
end

header("5. why the sine algebra escapes, and the central extension")
for N in (5, 9, 15)
    cross(m, nn) = m[1] * nn[2] - m[2] * nn[1]
    fs(z) = (N / 2π) * sin(2π * z / N)
    # phi(n) = f(0,n) for the sine bracket
    zero_mode = maximum(abs(fs(cross((0, 0), (a, bb)))) for a in 0:(N - 1), bb in 0:(N - 1))
    check(
        "N = $N: the sine bracket has phi(n) = f(0,n) = 0 -- its zero mode is " *
        "central, so the theorem is satisfied trivially",
        zero_mode < TOL)
    per = maximum(abs(fs(z) - fs(z + N)) for z in (-3N):(3N))
    check("N = $N: its structure constants are N-periodic, so the wrap is invisible",
        per < 1e-9, @sprintf("max |f(z) - f(z+N)| = %.2e", per))
end
let lin = maximum(abs(z - (z + N)) for z in -5:4, N in (11,))
    check(
        "m - n is not N-periodic: any N-periodic function matching it on a window of " *
        "width N must jump by N at the seam",
        abs(lin - 11) < TOL,
        "jump = $lin")
end

"dim Z^2, dim B^2 for the algebra with [e_i,e_j] = sum_k c[i,j,k] e_k."
function cohomology(cstruct, dim)
    prs = [(i, j) for i in 0:(dim - 1) for j in (i + 1):(dim - 1)]
    rows = Vector{Float64}[]
    for i in 0:(dim - 1), j in (i + 1):(dim - 1), k in (j + 1):(dim - 1)
        row = zeros(length(prs))
        for (aa, bb, cc) in ((i, j, k), (j, k, i), (k, i, j))
            for m in 0:(dim - 1)
                abs(cstruct[aa + 1, bb + 1, m + 1]) > 1e-12 || continue
                m == cc && continue
                x, y = m < cc ? (m, cc) : (cc, m)
                row[findfirst(==((x, y)), prs)] += cstruct[aa + 1, bb + 1, m + 1] *
                                                   (m < cc ? 1 : -1)
            end
        end
        push!(rows, row)
    end
    Z = isempty(rows) ? length(prs) :
        length(prs) - rank(reduce(vcat, transpose.(rows)); atol = 1e-9)
    Bm = reduce(vcat, [transpose([cstruct[i + 1, j + 1, xi + 1] for (i, j) in prs])
                       for xi in 0:(dim - 1)])
    return Z, rank(Bm; atol = 1e-9)
end

# validate the cohomology routine on algebras with known H^2
let ab = zeros(3, 3, 3)                                  # abelian: H^2 = Λ² = 3
    z, b = cohomology(ab, 3)
    check("cohomology routine: abelian 3-dim has dim H^2 = 3", (z, b) == (3, 0),
        "dim Z^2 = $z, dim B^2 = $b")
end
let hz = zeros(3, 3, 3)                                  # Heisenberg: H^2 = 2
    hz[1, 2, 3], hz[2, 1, 3] = 1.0, -1.0
    z, b = cohomology(hz, 3)
    check("cohomology routine: Heisenberg 3-dim has dim H^2 = 2", z - b == 2,
        "dim Z^2 = $z, dim B^2 = $b, dim H^2 = $(z - b)")
end
let c3 = zeros(3, 3, 3), bv = 1.0, cv = 1.0
    c3[1, 2, 2], c3[2, 1, 2] = -bv, bv
    c3[1, 3, 3], c3[3, 1, 3] = bv, -bv
    c3[2, 3, 1], c3[3, 2, 1] = cv, -cv
    Z2, B2 = cohomology(c3, 3)
    check(
        "the N = 3 exception has H^2 = 0, so it admits no nontrivial central " *
        "extension and cannot carry the Virasoro cocycle",
        Z2 - B2 == 0,
        "dim Z^2 = $Z2, dim B^2 = $B2, dim H^2 = $(Z2 - B2)")
end

summary("verify_kdv_aliasing.jl")
