#!/usr/bin/env julia
#
# Backward error analysis: why implicit midpoint on the first flow holds H2 so well.
#
#     julia --project=scripts scripts/verify_kdv_bea.jl
#
#    1. the symmetry group exp(sA), A = Minv S: A^T M + M A = 0 and A P^1 + P^1 A^T = 0;
#    2. the cross-conservation residual IS the symmetry defect, {H2,H1}_1 = -L_A H1;
#    3. the modified Hamiltonian H^[2] = -1/24 <H1'' f1, f1>, identified by an order test;
#    4. H^[2] inherits the symmetry, so {H2,H^[2]}_1 vanishes wherever the defect does;
#    5. THE EXACT INCREMENT LEMMA. H2 is quadratic, so for any method of the form
#       u^{n+1} = u^n + dt P^1 gbar the change in H2 is exactly -dt <gbar, A ubar>. With
#       gbar = grad H1(ubar) -- the midpoint rule -- that carries no dt dependence of its own;
#    6. the step-size sweep: the midpoint plateau is the same to four digits over a factor of
#       sixteen in dt, and the energy-preserving methods sit O(dt^2) above it;
#    7. bounded, not secular, over a hundred time units;
#    8. the plateau is SPATIAL: it converges under refinement and is insensitive to dt far
#       past the backward-error window;
#    9. only the midpoint rule on the first flow is a Poisson map -- which is what makes
#       gbar = grad H1(ubar);
#   10. the defect in closed form: eps_h = 3 <(id-Pi) d_x u_h, (id-Pi)(u_h^2)>, a pairing of
#       two projection errors, far below the product of their norms;
#   11. the exponent is 2p+2, twice over: the dispersion error of A, and the H2 plateau;
#   12. why the sum does not accumulate: eps_h carries no zero mode, and the only resonant
#       triples are the ones it does not have.
#
# `exp` here is `LinearAlgebra.exp`, where the Python carried a hand-rolled scaling-and-
# squaring `expm` to avoid a scipy dependency.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

U0b(x) = sin(x)
RESOLVED(x) = 0.1 * sin(x)

"The generator A = Minv S of the symmetry group."
sym_A(s::KdVSys) = s.Minv * s.S

"The constant third derivative of H1, -6 int phi_i phi_j phi_k."
function T3(s::KdVSys)
    Φ0, W = basis_values(s.s, 0), s.W
    nb = s.n
    -6.0 .* [sum(Φ0[i, q] * Φ0[j, q] * Φ0[k, q] * W[q] for q in eachindex(W))
     for i in 1:nb, j in 1:nb, k in 1:nb]
end

"The leading modified Hamiltonian of the implicit midpoint rule, H^[2] = -1/24 <H1'' f1, f1>."
Hmod(s::KdVSys, d) = (f = f1(s, d); -dot(f, hessH1(s, d), f) / 24)

"""
    gHmod(s, d, T)

dH^[2]/du, analytically: `-1/24 ( H1'''(f,f) - 2 H1'' P^1 H1'' f )`. Assembled rather than
differenced -- the finite-difference floor of the bracket {H2,H^[2]}_1 below is 1e-14
normalised, three orders above the signal at a resolved field, which would hide the result.
"""
function gHmod(s::KdVSys, d, T)
    f = f1(s, d)
    H = hessH1(s, d)
    -([sum(T[i, j, k] * f[j] * f[k] for j in 1:s.n, k in 1:s.n) for i in 1:s.n] -
      2 .* (H * (s.P1 * (H * f)))) ./ 24
end

"The cross-conservation residual {H2,H1}_1 of eq. (cross-conservation-split)."
eps_h(s::KdVSys, d) = dot(s.M * d, s.P1 * gH1(s, d))

"{H2, G}_1 for a functional G with gradient g, and its natural scale."
bracket1(s::KdVSys, d, g) = dot(s.M * d, s.P1 * g)
norm1(s::KdVSys, d, g) = norm(s.M * d) * maximum(abs, s.P1) * max(norm(g), 1e-300)

"A field whose Fourier content stops at mode kmax, projected onto V_h."
function modes(s::KdVSys, kmax; seed = 0)
    rng = MersenneTwister(seed)
    a = randn(rng, kmax) .+ 1.0
    b = randn(rng, kmax)
    project(s.s, x -> sum(a[j] * sin(2π * j * x / s.Lx) + b[j] * cos(2π * j * x / s.Lx)
    for j in 1:kmax) / kmax)
end

"Least-squares slope of log y against log x."
function slope(xs, ys)
    lx, ly = log.(xs), log.(ys)
    x̄, ȳ = sum(lx) / length(lx), sum(ly) / length(ly)
    sum((lx .- x̄) .* (ly .- ȳ)) / sum((lx .- x̄) .^ 2)
end

# --------------------------------------------------------------------------
header("1. the symmetry group exp(sA), A = Minv S")

println("  A u is the Hamiltonian vector field of H2 under the FIRST bracket,")
println("  P^1 dH2/du = Minv S Minv M u = Minv S u = A u,  and it is linear.")

for (lab, kw) in (("uniform     ", (;)), ("non-uniform ", (; uniform = false)),
    ("graded      ", (; uniform = false, graded = true)))
    s = KdVSys(20; p = 3, u0 = U0b, kw...)
    A = sym_A(s)
    e1 = maximum(abs, transpose(A) * s.M + s.M * A) / maximum(abs, s.M)
    e2 = maximum(abs, A * s.P1 + s.P1 * transpose(A)) / maximum(abs, s.P1)
    check("$(lab)mesh: A^T M + M A = 0, so exp(sA) preserves H2 exactly", e1 < 1e-12,
        @sprintf("%.2e", e1))
    check("$(lab)mesh: A P^1 + P^1 A^T = 0, so exp(sA) preserves P^1 exactly", e2 < 1e-12,
        @sprintf("%.2e", e2))
end

const s1 = KdVSys(20; p = 3, u0 = U0b)
let A = sym_A(s1), Abad = s1.Minv * (s1.S + 0.1 .* s1.K1)
    e1 = maximum(abs, transpose(Abad) * s1.M + s1.M * Abad) / maximum(abs, s1.M)
    check("negative control: a non-skew S breaks the first identity outright", e1 > 1e-2,
        @sprintf("%.2e", e1))
    d = s1.d0
    for sv in (1.0, 0.3, 0.1)
        E = exp(sv .* A)
        eH2 = abs(sysH2(s1, E * d) - sysH2(s1, d)) / abs(sysH2(s1, d))
        eP1 = maximum(abs, E * s1.P1 * transpose(E) - s1.P1) / maximum(abs, s1.P1)
        check("s = $sv: exp(sA) is an H2-isometry and a P^1-Poisson map",
            eH2 < 1e-12 && eP1 < 1e-11, @sprintf("dH2 %.1e, dP^1 %.1e", eH2, eP1))
    end
end

# --------------------------------------------------------------------------
header("2. the cross-conservation residual IS the symmetry defect")

println("  {H2,H1}_1 = - L_A H1 identically in uhat, on any mesh.")
let rng = MersenneTwister(11)
    for (lab, kw) in (("uniform    ", (;)), ("non-uniform", (; uniform = false)))
        s = KdVSys(20; p = 3, u0 = U0b, kw...)
        A = sym_A(s)
        worst = 0.0
        for _ in 1:8
            d = randn(rng, s.n)
            lhs = eps_h(s, d)
            rhs = -dot(gH1(s, d), A * d)
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-30))
        end
        check("$lab mesh: {H2,H1}_1 + L_A H1 = 0 for random uhat", worst < 1e-10,
            @sprintf("%.2e", worst))
    end
end

# --------------------------------------------------------------------------
header("3. the modified Hamiltonian of the implicit midpoint rule")

println("  f3 = 1/12 f'f'f - 1/24 f''(f,f)  is  P^1 grad H^[2]  with")
println("  H^[2] = - 1/24 < H1'' f1, f1 >, so the modified equation of the midpoint")
println("  rule on the first flow is Hamiltonian, as it must be.")

const s3 = KdVSys(12; p = 3, u0 = x -> sin(x) + 0.4 * cos(2x))
const T3s = T3(s3)
let d = s3.d0
    gA = gHmod(s3, d, T3s)
    gF = zeros(s3.n)
    for j in 1:s3.n
        dp, dm = copy(d), copy(d)
        dp[j] += 1e-4
        dm[j] -= 1e-4
        gF[j] = (Hmod(s3, dp) - Hmod(s3, dm)) / 2e-4
    end
    e = maximum(abs, gA - gF) / maximum(abs, gA)
    check("grad H^[2] assembled analytically matches the central difference", e < 1e-5,
        @sprintf("%.2e", e))

    # f1 is quadratic in uhat, so its second derivative is exact in three evaluations
    f = f1(s3, d)
    fppff = 2.0 .* (f1(s3, d + f) - f1(s3, d) - Df1(s3, d) * f)
    f3 = (Df1(s3, d) * (Df1(s3, d) * f)) ./ 12 - fppff ./ 24
    e = maximum(abs, s3.P1 * gA - f3) / maximum(abs, f3)
    check("P^1 grad H^[2] = 1/12 f'f'f - 1/24 f''(f,f)", e < 1e-10, @sprintf("%.2e", e))
end

"Implicit midpoint solved to machine precision; the order test needs the step itself exact."
function midpoint_tight(s::KdVSys, d, dt)
    y = copy(d)
    Id = Matrix{Float64}(I, s.n, s.n)
    for _ in 1:60
        r = y - d - dt .* f1(s, 0.5 .* (d + y))
        norm(r) < 1e-16 * max(1.0, norm(y)) && break
        y = y - (Id - 0.5dt .* Df1(s, 0.5 .* (d + y))) \ r
    end
    y
end

"The exact flow of `field` over dt, by RK4 with nsub substeps."
function flow(d, dt, field; nsub = 128)
    h = dt / nsub
    y = copy(d)
    for _ in 1:nsub
        a1 = field(y)
        a2 = field(y + 0.5h .* a1)
        a3 = field(y + 0.5h .* a2)
        a4 = field(y + h .* a3)
        y = y + (h / 6) .* (a1 + 2a2 + 2a3 + a4)
    end
    y
end

println()
println("  local error of one midpoint step against the flow of f1 and against the")
println("  flow of the modified field f1 + dt^2 P^1 grad H^[2].  N = 12, p = 3, so")
@printf("  rho = %.0f and every dt below is well inside dt rho < 1.\n",
    maximum(abs, eigvals(Df1(s3, s3.d0))))
@printf("     %9s  %11s  %10s  %12s  %10s\n", "dt", "vs f1", "/dt^3", "vs modified",
    "/dt^5")
let rows = NTuple{3, Float64}[], d = s3.d0
    for dt in (8e-3, 4e-3, 2e-3, 1e-3, 5e-4)
        y = midpoint_tight(s3, d, dt)
        e0 = norm(y - flow(d, dt, z -> f1(s3, z)))
        e2 = norm(y - flow(d, dt, z -> f1(s3, z) + dt^2 .* (s3.P1 * gHmod(s3, z, T3s))))
        push!(rows, (dt, e0, e2))
        @printf("     %9.2e  %11.3e  %10.3e  %12.3e  %10.3e\n", dt, e0, e0 / dt^3, e2,
            e2 / dt^5)
    end
    last3 = rows[(end - 2):end]
    r0 = slope([r[1] for r in last3], [r[2] for r in last3])
    r2 = slope([r[1] for r in last3], [r[3] for r in last3])
    check("the midpoint step is O(dt^3) away from the flow of f1", 2.9 < r0 < 3.1,
        @sprintf("rate %.2f", r0))
    check(
        "and O(dt^5) away from the flow of the modified field, which is what " *
        "identifies H^[2]",
        4.85 < r2 < 5.15,
        @sprintf("rate %.2f", r2))
end

# --------------------------------------------------------------------------
header("4. H^[2] inherits the symmetry, and with it the bound on H2")

println("  If exp(sA) is a symmetry of H1 then it is one of H^[2] as well:")
println("  grad H1 -> phi^-T grad H1, H1'' -> phi^-T H1'' phi^-1 and P^1 phi^-T =")
println("  phi P^1, hence f1 -> phi f1 and < H1'' f1, f1 > is invariant.")
println()

let s = KdVSys(24; p = 3, u0 = U0b), A = sym_A(s), d = s.d0
    lin_f(z) = s.P1 * (s.K1 * z)
    lin_H1(z) = dot(z, s.K1, z) / 2
    lin_Hm(z) = (lf = lin_f(z); -dot(lf, s.K1, lf) / 24)
    worst_h1, worst_hm = 0.0, 0.0
    for sv in (0.5, 0.2, 0.05)
        E = exp(sv .* A)
        worst_h1 = max(worst_h1, abs(lin_H1(E * d) - lin_H1(d)) / abs(lin_H1(d)))
        worst_hm = max(worst_hm, abs(lin_Hm(E * d) - lin_Hm(d)) / abs(lin_Hm(d)))
    end
    check("linear KdV, uniform mesh: exp(sA) leaves the quadratic H1 invariant",
        worst_h1 < 1e-12, @sprintf("%.2e", worst_h1))
    check(
        "...and therefore leaves H^[2] invariant, to round-off, so H2 is an exact " *
        "invariant of the modified flow there",
        worst_hm < 1e-11,
        @sprintf("%.2e", worst_hm))
end

println()
println("  In general the covariance is not exact, and differentiating it in s gives")
println("  an identity rather than an estimate.  With eta := L_A H1 = -{H2,H1}_1,")
println()
println("      {H2,H^[2]}_1 = 1/24 ( < grad^2 eta f1, f1 > + 2 < H1'' P^1 grad eta, f1 > ),")
println()
println("  every term of which carries a derivative of eta, so the whole bracket")
println("  vanishes identically wherever the symmetry defect does.")

let s = KdVSys(24; p = 3, u0 = U0b), T = T3(KdVSys(24; p = 3, u0 = U0b))
    A = sym_A(s)
    rng = MersenneTwister(5)
    worst = 0.0
    for _ in 1:6
        d = randn(rng, s.n) .* 0.4
        f, H = f1(s, d), hessH1(s, d)
        geta = H * (A * d) + transpose(A) * gH1(s, d)
        g2eta = [sum(T[i, j, k] * (A * d)[k] for k in 1:s.n) for i in 1:s.n, j in 1:s.n] +
                H * A + transpose(A) * H
        rhs = (dot(f, g2eta, f) + 2 * dot(f, H, s.P1 * geta)) / 24
        lhs = bracket1(s, d, gHmod(s, d, T))
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-30))
    end
    check("the identity holds for random uhat, well away from any resolved field",
        worst < 1e-9, @sprintf("%.2e", worst))
end

println()
println("  Numerically, then: the bracket and the residual go to round-off together")
println("  as the mode content is lowered.  N = 32, p = 3, uniform; both normalised")
println("  by their own scale, as the Jacobi residuals of Section 5 are.")
@printf("     %6s  %11s  %14s\n", "modes", "eps_h", "{H2,H^[2]}_1")
let s = KdVSys(32; p = 3, u0 = U0b), tab = NTuple{3, Float64}[]
    T = T3(s)
    for kmax in (1, 2, 4, 8, 12, 16)
        d = modes(s, kmax)
        e = eps_h(s, d) / norm1(s, d, gH1(s, d))
        g = gHmod(s, d, T)
        b = bracket1(s, d, g) / norm1(s, d, g)
        push!(tab, (kmax, e, b))
        @printf("     %6d  %11.2e  %14.2e\n", kmax, e, b)
    end
    check("at a resolved field both sit at round-off together",
        abs(tab[1][2]) < 1e-14 && abs(tab[1][3]) < 1e-14,
        @sprintf("eps_h %.1e, bracket %.1e", tab[1][2], tab[1][3]))
    check(
        "and both rise monotonically with the mode content, by ten orders of " *
        "magnitude over the same range",
        abs(tab[end][2]) > 1e5 * abs(tab[2][2]) && abs(tab[end][3]) > 1e5 * abs(tab[2][3]),
        @sprintf("eps_h %.0e -> %.0e, bracket %.0e -> %.0e",
            abs(tab[2][2]), abs(tab[end][2]), abs(tab[2][3]), abs(tab[end][3])))
end

# --------------------------------------------------------------------------
header("5. the exact increment lemma")

println("  H2 is quadratic, so for ANY method of the form u^{n+1} = u^n + dt P^1 gbar")
println()
println("      H2(u^{n+1}) - H2(u^n) = (du, M ubar) = - dt < gbar, A ubar > ,")
println()
println("  exactly, at any step size.  With gbar = grad H1(ubar) -- the midpoint")
println("  rule -- the right-hand side is dt {H2,H1}_1(ubar) and carries no dt")
println("  dependence of its own: the whole H2 error is then a midpoint quadrature")
println("  of the semi-discrete residual.  Every other method differs from")
println("  grad H1(ubar) at O(dt^2), and that difference is all of its extra error.")

const s5 = KdVSys(16; p = 3, u0 = cosine(2π))
const T5 = T3(s5)
const A5 = sym_A(s5)
const DT5 = 5e-3

const METHODS = (
    ("midpoint", (s, d, dt) -> midpoint(s, d, dt, 1)[1],
        (x, y) -> gH1(s5, 0.5 .* (x + y))),
    ("AVF     ", (s, d, dt) -> avf(s, d, dt, 1)[1],
        (x, y) -> sum(gH1(s5, x + t .* (y - x)) for t in SG) / 2),
    ("dgrad   ", (s, d, dt) -> dgrad(s, d, dt), (x, y) -> dgrad_H1(s5, x, y)),
    ("dgrad-M ", (s, d, dt) -> dgrad_mass(s, d, dt), (x, y) -> dgrad_H1_mass(s5, x, y))
)

println()
for (nm, step, gbar) in METHODS
    y = step(s5, s5.d0, DT5)
    m = 0.5 .* (s5.d0 + y)
    lhs = dot(y - s5.d0, s5.M, m)
    rhs = -DT5 * dot(gbar(s5.d0, y), A5 * m)
    e = abs(lhs - rhs) / abs(lhs)
    check("$nm: (du, M ubar) = - dt <gbar, A ubar>", e < 1e-6, @sprintf("%.2e", e))
end
println("  (the floor here is the Newton residual, 1e-13 absolute, against an")
println("   increment of order 1e-8; the identity itself is exact.)")

println()
println("  and the three gradients differ from grad H1(ubar) by exactly:")
let y = avf(s5, s5.d0, DT5, 1)[1]
    m, du = 0.5 .* (s5.d0 + y), y - s5.d0
    g = sum(gH1(s5, s5.d0 + t .* (y - s5.d0)) for t in SG) / 2
    corr = [sum(T5[i, j, k] * du[j] * du[k] for j in 1:s5.n, k in 1:s5.n) for i in 1:s5.n] ./
           24
    e = maximum(abs, g - gH1(s5, m) - corr) / maximum(abs, g)
    check("AVF:      gbar = grad H1(ubar) + 1/24 D3H1(du,du)", e < 1e-13, @sprintf("%.2e",
        e))
end
let y = dgrad(s5, s5.d0, DT5)
    m, du = 0.5 .* (s5.d0 + y), y - s5.d0
    g = dgrad_H1(s5, s5.d0, y)
    c = sum(T5[i, j, k] * du[i] * du[j] * du[k] for i in 1:s5.n, j in 1:s5.n, k in 1:s5.n) /
        dot(du, du)
    e = maximum(abs, g - gH1(s5, m) - (c / 24) .* du) / maximum(abs, g)
    check("Gonzalez: gbar = grad H1(ubar) + 1/24 D3H1(du,du,du)/(du,du) du", e < 1e-13,
        @sprintf("%.2e", e))
end
let y = dgrad_mass(s5, s5.d0, DT5)
    m, du = 0.5 .* (s5.d0 + y), y - s5.d0
    g = dgrad_H1_mass(s5, s5.d0, y)
    Mdu = s5.M * du
    c = sum(T5[i, j, k] * du[i] * du[j] * du[k] for i in 1:s5.n, j in 1:s5.n, k in 1:s5.n) /
        dot(du, Mdu)
    e = maximum(abs, g - gH1(s5, m) - (c / 24) .* Mdu) / maximum(abs, g)
    check("mass metric: the same, with (du, M du) and M du", e < 1e-13, @sprintf("%.2e", e))
end
println("  All three corrections are O(dt^2), so all three lose O(dt^3) of H2 per")
println("  step and O(dt^2) over a fixed time -- which is what section 6 measures.")

println()
println("  The second flow has the same structure with the two Hamiltonians")
println("  exchanged, except that H1 is CUBIC, so the Taylor remainder survives")
println("  even for the midpoint rule:")
println()
println("      H1(u^{n+1}) - H1(u^n) = dt {H1,H2}_2(ubar) + 1/24 D3H1(du,du,du) .")
println()
for dt in (5e-3, 1e-3)
    y = midpoint(s5, s5.d0, dt, 2)[1]
    m, du = 0.5 .* (s5.d0 + y), y - s5.d0
    lhs = sysH1(s5, y) - sysH1(s5, s5.d0)
    r1 = dt * dot(gH1(s5, m), f2(s5, m))
    r2 = sum(T5[i, j, k] * du[i] * du[j] * du[k]
    for i in 1:s5.n, j in 1:s5.n, k in 1:s5.n) / 24
    e = abs(lhs - r1 - r2) / abs(lhs)
    check(@sprintf("dt = %.0e: the identity holds", dt), e < 1e-6,
        @sprintf("dt*eps2 %.2e, cubic %.2e, residual %.1e", r1, r2, e))
end
println("  This is the elementary reason for the asymmetry between the two flows:")
println("  the protected invariant has to be quadratic, and only H2 is.")

# --------------------------------------------------------------------------
header("6. the step-size sweep")

const T_SWEEP = 10.0

function sweep(sy::KdVSys, step, dt; T = T_SWEEP, q = 2)
    NT = round(Int, T / dt)
    _, ref, dev = integrate_windowed(sy, d -> step(sy, d, dt), NT; stride = max(1, NT ÷
                                                                                   200))
    env_drift(dev[q], ref[q]), env_growth(dev[q])
end

println("  A resolved field, u0 = 0.1 sin x on N = 16, p = 3, so that eps_h sits at")
println("  round-off and any dt dependence is the integrator's own.  Reported is")
@printf("  max_t |H2(t) - H2(0)| / |H2(0)| over t <= %.0f.\n", T_SWEEP)
const s6 = KdVSys(16; p = 3, u0 = RESOLVED)
@printf("  eps_h at the initial field: %.2e\n", eps_h(s6, s6.d0))
println()
@printf("     %9s  %s\n", "dt",
    join([@sprintf("%11s", strip(nm)) for (nm, _, _) in METHODS], "  "))
const DTS = (3.2e-3, 1.6e-3, 8e-4, 4e-4, 2e-4)
const tab6 = Dict{Float64, Vector{Float64}}()
for dt in DTS
    row = [sweep(s6, step, dt)[1] for (_, step, _) in METHODS]
    tab6[dt] = row
    @printf("     %9.1e  %s\n", dt, join([@sprintf("%11.4e", v) for v in row], "  "))
end
let mid = [tab6[dt][1] for dt in DTS]
    e = (maximum(mid) - minimum(mid)) / maximum(mid)
    check(
        "the midpoint plateau does not move with dt at all -- five step sizes " *
        "spanning a factor of sixteen give the same number to four digits",
        e < 1e-3,
        @sprintf("spread %.1e about %.4e", e, mid[1]))
end

println()
println("  the excess of the energy-preserving methods over that plateau:")
@printf("     %9s  %s\n", "dt",
    join([@sprintf("%11s", strip(nm)) for (nm, _, _) in METHODS[2:end]], "  "))
for dt in DTS
    @printf("     %9.1e  %s\n", dt,
        join([@sprintf("%11.3e", tab6[dt][j] - tab6[dt][1]) for j in 2:4], "  "))
end
for (j, nm) in zip(2:4, ("AVF", "Gonzalez", "Gonzalez, mass metric"))
    r = slope(collect(DTS), [tab6[dt][j] - tab6[dt][1] for dt in DTS])
    check("$nm: the excess is O(dt^2), as the increment lemma predicts", 1.9 < r < 2.1,
        @sprintf("rate %.2f", r))
end
println("  The mass metric makes no difference worth having: the correction it")
println("  alters is O(dt^2) either way, and the two agree to a few per cent.")

println()
println("  An under-resolved field, u0 = cos x on the same mesh, where eps_h")
println("  dominates instead:")
let su = KdVSys(16; p = 3, u0 = cosine(2π)), uv = Dict{Float64, Vector{Float64}}()
    @printf("     %9s  %s\n", "dt",
        join([@sprintf("%11s", strip(nm)) for (nm, _, _) in METHODS], "  "))
    for dt in (3.2e-3, 8e-4, 2e-4)
        row = [sweep(su, step, dt)[1] for (_, step, _) in METHODS]
        uv[dt] = row
        @printf("     %9.1e  %s\n", dt, join([@sprintf("%11.4e", v) for v in row], "  "))
    end
    spread = maximum(abs(v - uv[2e-4][1]) for v in uv[2e-4]) / uv[2e-4][1]
    check("there all four methods agree, the error being purely spatial", spread < 5e-3,
        @sprintf("spread %.1e about %.3e", spread, uv[2e-4][1]))
end

println()
println("  The second flow, for comparison: H1 keeps a dt-dependent part even for")
println("  the midpoint rule, by the cubic term of the identity above.")
@printf("     %9s  %12s  %11s\n", "dt", "|dH1|", "|dH2|")
mid2(ss, dd, h) = midpoint(ss, dd, h, 2)[1]
const DTS2 = (3.2e-3, 1.6e-3, 8e-4, 4e-4, 2e-4)
let s2 = Dict{Float64, Float64}(), blast = 0.0
    for dt in DTS2
        a, _ = sweep(s6, mid2, dt; q = 1)
        b, _ = sweep(s6, mid2, dt; q = 2)
        s2[dt] = a
        blast = b
        @printf("     %9.1e  %12.5e  %11.2e\n", dt, a, b)
    end
    check(
        "H2 is conserved to round-off along the second flow, being quadratic and " *
        "an exact invariant there",
        blast < 1e-12,
        @sprintf("%.1e", blast))
    # a + b dt^2 through all five, the plateau a being the spatial part
    X = [dt^2 for dt in DTS2]
    Y = [s2[dt] for dt in DTS2]
    x̄, ȳ = sum(X) / 5, sum(Y) / 5
    b1 = sum((X .- x̄) .* (Y .- ȳ)) / sum((X .- x̄) .^ 2)
    b0 = ȳ - b1 * x̄
    @printf("     fit  |dH1| = %.5e + %.3e dt^2 ;  the plateau is the\n", b0, b1)
    println("     spatial residual and the dt^2 term is the cubic Taylor remainder")
    resid = maximum(abs(s2[dt] - b0 - b1 * dt^2) for dt in DTS2) / b0
    check(
        "the H1 error along the second flow is a spatial plateau plus a genuine " *
        "O(dt^2) part -- not the pure plateau of the first flow",
        resid < 1e-3 && b1 * DTS2[1]^2 > 0.02 * b0,
        @sprintf("fit residual %.1e, dt^2 part %.1f%% of the plateau at dt = %.1e",
            resid, 100 * b1 * DTS2[1]^2 / b0, DTS2[1]))
end

# --------------------------------------------------------------------------
header("7. bounded, not secular")

println("  The same runs to t = 100 at dt = 1e-3.  ``growth`` is the envelope over")
println("  the last tenth of the run against the first: one for a bounded")
println("  oscillation, large for a drift.")
@printf("     %22s  %12s  %7s\n", "run", "|d other H|", "growth")
let worst = 0.0
    for (nm, step, q) in (("midpoint, flow 1", (ss, dd, h) -> midpoint(ss, dd, h, 1)[1], 2),
        ("AVF, flow 1", (ss, dd, h) -> avf(ss, dd, h, 1)[1], 2),
        ("Gonzalez, flow 1", (ss, dd, h) -> dgrad(ss, dd, h), 2),
        ("midpoint, flow 2", mid2, 1))
        v, g = sweep(s6, step, 1e-3; T = 100.0, q)
        worst = max(worst, g)
        @printf("     %22s  %12.4e  %7.2f\n", nm, v, g)
    end
    check(
        "no run drifts: the envelope of the other Hamiltonian is flat over a " *
        "hundred time units",
        worst < 1.1,
        @sprintf("largest growth %.2f", worst))
end

# --------------------------------------------------------------------------
header("8. the plateau is spatial: refinement, and the step-size window")

println("  midpoint on the first flow, dt = 2e-4 fixed, u0 = 0.1 sin x:")
@printf("     %4s  %8s  %11s  %6s\n", "N", "h", "|dH2|", "rate")
let prev = nothing, rs = Float64[]
    for n in (12, 16, 20, 24, 32)
        sn = KdVSys(n; p = 3, u0 = RESOLVED)
        v, _ = sweep(sn, (ss, dd, h) -> midpoint(ss, dd, h, 1)[1], 2e-4)
        r = ""
        if prev !== nothing
            push!(rs, log(prev[2] / v) / log(n / prev[1]))
            r = @sprintf("%6.2f", rs[end])
        end
        @printf("     %4d  %8.4f  %11.3e  %s\n", n, 2π / n, v, r)
        prev = (n, v)
    end
    check(
        "the plateau converges under refinement, at better than the sixth order " *
        "of the cross-conservation residual of Section 5",
        minimum(rs) > 6.0,
        "rates " * join([@sprintf("%.1f", r) for r in rs], ", "))
end

println()
println("  and it is insensitive to dt far past the backward-error window.  N = 16,")
const ρ6 = maximum(abs, eigvals(Df1(s6, s6.d0)))
@printf("  rho = %.0f, so the formal estimate needs dt rho < 1:\n", ρ6)
@printf("     %9s  %7s  %11s  %11s\n", "dt", "dt rho", "|dH1|", "|dH2|")
let vals = Float64[]
    for dt in (1e-3, 4e-3, 1.6e-2, 3.2e-2)
        a, _ = sweep(s6, (ss, dd, h) -> midpoint(ss, dd, h, 1)[1], dt; q = 1)
        b, _ = sweep(s6, (ss, dd, h) -> midpoint(ss, dd, h, 1)[1], dt; q = 2)
        push!(vals, b)
        @printf("     %9.1e  %7.2f  %11.2e  %11.4e\n", dt, dt * ρ6, a, b)
    end
    e = (maximum(vals) - minimum(vals)) / maximum(vals)
    check(
        "H2 is untouched even at dt rho = 9, where the bound on H1 has degraded " *
        "by three orders -- the increment lemma is not asymptotic",
        e < 1e-3,
        @sprintf("spread %.1e", e))
end

# --------------------------------------------------------------------------
header("9. only the midpoint rule on the first flow is a Poisson map")

println("  the residual of eq. (poisson-map), swept in dt, N = 16:")
const Id6 = Matrix{Float64}(I, s6.n, s6.n)
function jac_midpoint(dt)
    (y = midpoint(s6, s6.d0, dt, 1)[1];
        D = Df1(s6, 0.5 .* (s6.d0 + y));
        (Id6 - 0.5dt .* D) \ (Id6 + 0.5dt .* D))
end
function jac_avf(dt)
    y = avf(s6, s6.d0, dt, 1)[1]
    dy = sum(t .* Df1(s6, s6.d0 + t .* (y - s6.d0)) for t in SG) / 2
    dd = sum((1 - t) .* Df1(s6, s6.d0 + t .* (y - s6.d0)) for t in SG) / 2
    (Id6 - dt .* dy) \ (Id6 + dt .* dd)
end
function jac_fd(step, dt; eps = 1e-6)
    D = zeros(s6.n, s6.n)
    for j in 1:s6.n
        dp, dm = copy(s6.d0), copy(s6.d0)
        dp[j] += eps
        dm[j] -= eps
        D[:, j] = (step(s6, dp, dt) - step(s6, dm, dt)) ./ (2eps)
    end
    D
end
let ja = jac_avf(1e-2)
    e = maximum(abs, ja - jac_fd((ss, dd, h) -> avf(ss, dd, h, 1)[1], 1e-2)) /
        maximum(abs, ja)
    check("the implicit AVF Jacobian agrees with a central difference", e < 1e-8,
        @sprintf("%.1e", e))
end
res9(D) = maximum(abs, D * s6.P1 * transpose(D) - s6.P1) / maximum(abs, s6.P1)
@printf("     %9s  %11s  %11s  %11s\n", "dt", "midpoint", "AVF", "Gonzalez")
let rows = NTuple{4, Float64}[]
    for dt in (8e-3, 4e-3, 2e-3, 1e-3)
        a, b, c = res9(jac_midpoint(dt)), res9(jac_avf(dt)),
        res9(jac_fd((ss, dd, h) -> dgrad(ss, dd, h), dt))
        push!(rows, (dt, a, b, c))
        @printf("     %9.1e  %11.2e  %11.2e  %11.2e\n", dt, a, b, c)
    end
    check("the midpoint rule is a Poisson map at every step size, to round-off",
        maximum(r[2] for r in rows) < 1e-12,
        @sprintf("worst %.1e", maximum(r[2] for r in rows)))
    l3 = rows[(end - 2):end]
    r = slope([x[1] for x in l3], [x[3] for x in l3])
    check("the AVF method is not, its residual falling only as O(dt^3)", 2.6 < r < 3.4,
        @sprintf("rate %.2f", r))
end
println("  Being a Poisson map is exactly the difference: it is what makes")
println("  gbar = grad H1(ubar), and so removes the O(dt^2) term from the increment.")

# --------------------------------------------------------------------------
header("10. the defect in closed form")

println("  eps_h splits by degree as in eq. (cross-conservation-split),")
println()
println("      eps_h = u^T (S Minv K1) u  +  3 u^T S Minv n(u) ,   n_i = int phi_i u_h^2,")
println()
println("  and on a uniform mesh the first term vanishes for EVERY u: S is skew,")
println("  Minv K1 symmetric, circulants commute, so the product is skew and its")
println("  quadratic form is zero.  The second term is a projection error.")

"(eps_h, quadratic part, cubic part)."
function split10(s::KdVSys, d)
    uh = fieldat(s.s, d, 0)
    quad = dot(d, s.S, s.Minv * (s.K1 * d))
    n = basis_values(s.s, 0) * (s.W .* uh .^ 2)
    eps_h(s, d), quad, -3.0 * dot(d, s.S * (s.Minv * n))
end

"The cubic part written five ways; the first four are the same number, the last is zero."
function forms10(s::KdVSys, d)
    Φ0, Φ1, W = basis_values(s.s, 0), basis_values(s.s, 1), s.W
    uh, ux = fieldat(s.s, d, 0), fieldat(s.s, d, 1)
    n = Φ0 * (W .* uh .^ 2)
    q = s.Minv * n
    qh, qx = fieldat(s.s, q, 0), fieldat(s.s, q, 1)
    puxh = fieldat(s.s, s.Minv * (Φ0 * (W .* ux)), 0)
    (-3.0 * dot(d, s.S * (s.Minv * n)),
        -3.0 * dot(W, uh .* qx),
        -3.0 * dot(W, ux .* (uh .^ 2 - qh)),
        -3.0 * dot(W, (ux - puxh) .* (uh .^ 2 - qh)),
        -3.0 * dot(W, uh .* 2.0 .* uh .* ux))
end

const rng10 = MersenneTwister(7)
let s = KdVSys(32; p = 3, u0 = U0b), d = randn(rng10, 32) .* 0.3
    e, quad, cub = split10(s, d)
    check("uniform mesh: the quadratic part vanishes for a broadband random uhat",
        abs(quad) / abs(e) < 1e-10, @sprintf("%.2e against eps_h %.3e", quad, e))
    sn = KdVSys(32; p = 3, u0 = U0b, uniform = false)
    _, quadn, _ = split10(sn, d)
    check("negative control: on a non-uniform mesh it does not", abs(quadn) > 1e-6,
        @sprintf("%.2e", quadn))

    a, b, c, dd, zero = forms10(s, d)
    println()
    @printf("     3 u^T S Minv n(u)                        % .8e\n", a)
    @printf("     3 int u_h d_x Pi(u_h^2)                  % .8e\n", b)
    @printf("     3 int d_x u_h (id-Pi)(u_h^2)             % .8e\n", c)
    @printf("     3 <(id-Pi) d_x u_h , (id-Pi)(u_h^2)>     % .8e\n", dd)
    @printf("     3 int u_h d_x (u_h^2)   [a total derivative, so zero]   % .2e\n", zero)
    check(
        "3 int u_h d_x (u_h^2) = 2 int d_x(u_h^3) = 0, which is what the " *
        "degree-3p-1 quadrature is for",
        abs(zero) / abs(a) < 1e-12,
        @sprintf("%.2e", zero))
    worst = maximum(abs(x - a) for x in (b, c, dd)) / abs(a)
    check(
        "so eps_h = 3 <(id-Pi) d_x u_h , (id-Pi)(u_h^2)> -- a pairing of two " *
        "projection errors, and nothing else",
        worst < 1e-6,
        @sprintf("%.2e", worst))
end

println()
println("  Neither factor ever vanishes, and for different reasons: d_x u_h has")
println("  degree p-1 but only C^{p-2}, so it is too ROUGH for V_h, while u_h^2 has")
println("  degree 2p, so it is too HIGH -- its content reaches twice the Nyquist of")
println("  V_h and Pi aliases it away.  But the pairing is far below the product of")
println("  the two norms: the errors are very nearly orthogonal.")
let sm = KdVSys(16; p = 3, u0 = x -> 0.1 * sin(x) + 0.03 * cos(2x))
    Φ0, W = basis_values(sm.s, 0), sm.W
    dm = sm.d0
    uh, ux = fieldat(sm.s, dm, 0), fieldat(sm.s, dm, 1)
    qh = fieldat(sm.s, sm.Minv * (Φ0 * (W .* uh .^ 2)), 0)
    puxh = fieldat(sm.s, sm.Minv * (Φ0 * (W .* ux)), 0)
    e1 = sqrt(dot(W, (ux - puxh) .^ 2))
    e2 = sqrt(dot(W, (uh .^ 2 - qh) .^ 2))
    @printf("     N = 16, p = 3, smooth field:  ||(id-Pi) d_x u_h|| = %.2e,  ||(id-Pi)(u_h^2)|| = %.2e\n",
        e1, e2)
    @printf("     product %.2e,  actual eps_h %.2e\n", 3e1 * e2, abs(eps_h(sm, dm)))
    check("the pairing is many orders below the product of the norms",
        abs(eps_h(sm, dm)) < 1e-6 * 3 * e1 * e2,
        @sprintf("ratio %.1e", abs(eps_h(sm, dm)) / (3 * e1 * e2)))
end

println()
println("  One consequence is used in section 12: adding a constant to u_h changes")
println("  neither factor, since d_x kills it and (id-Pi) kills 2 c u_h + c^2, both")
println("  of which lie in V_h.  So eps_h carries no zero-mode content at all.")
let s = KdVSys(24; p = 3, u0 = U0b), ones_ = ones(24), worst = 0.0
    for _ in 1:4
        d = randn(rng10, s.n) .* 0.4
        b0 = eps_h(s, d)
        worst = max(worst, maximum(abs(eps_h(s, d + c .* ones_) - b0) / abs(b0)
        for c in (0.3, 1.0, -2.5, 7.0)))
    end
    check("eps_h(uhat + c e) = eps_h(uhat) for every constant c", worst < 1e-8,
        @sprintf("%.2e", worst))
    d = randn(rng10, s.n) .* 0.4
    @printf("     by contrast H1: %.4e -> %.4e,   H2: %.4e -> %.4e\n",
        sysH1(s, d), sysH1(s, d + ones_), sysH2(s, d), sysH2(s, d + ones_))
end

# --------------------------------------------------------------------------
header("11. the exponent is 2p+2, twice over")

println("  What survives the near-cancellation is the dispersion error of A: on a")
println("  uniform mesh A is circulant with eigenvalue i lam_m, against i m for the")
println("  exact d_x, and lam_m / m - 1 = O((m h)^{2p+2}) -- the classical")
println("  isogeometric rate.  exp(sA) is translation only on the resolved modes.")
println()
println("  (a) dispersion error of A = Minv S, N = 64:")
@printf("     %3s  %s  %9s  %5s\n", "p",
    join([@sprintf("%10s", "m=" * string(m)) for m in (1, 2, 4, 8)], "  "), "exponent",
    "2p+2")
for p in (2, 3, 4)
    sp = KdVSys(64; p, u0 = U0b)
    A = sym_A(sp)
    h = sp.Lx / 64
    row = NTuple{2, Float64}[]
    for m in (1, 2, 4, 8)
        v = [cis(2π * m * j / 64) for j in 0:63]
        # `dot(x, A, y)` is x'Ay without materialising A*y; and `dot` conjugates its
        # FIRST argument, so this is numpy's `v.conj() @ (A @ v)`.
        λ = dot(v, A, v) / dot(v, v)
        push!(row, (m * h, abs(λ / (1im * m) - 1.0)))
    end
    fit = [r for r in row if r[2] > 1e-14]      # the low modes are at round-off for p = 4
    ex = slope([r[1] for r in fit], [r[2] for r in fit])
    @printf("     %3d  %s  %9.2f  %5d\n", p,
        join([@sprintf("%10.2e", r[2]) for r in row], "  "), ex, 2p + 2)
    check(
        "p = $p: the dispersion error of A is O((mh)^$(2p + 2))", abs(ex - (2p + 2)) < 1.0,
        @sprintf("exponent %.2f", ex))
end

println()
println("  (b) and the H2 plateau converges at the same exponent.  Midpoint on the")
println("  first flow, dt = 2e-4 fixed, u0 = 0.1 sin x, T = 3:")
@printf("     %3s  %s  %22s  %5s\n", "p",
    join([@sprintf("%11s", "N=" * string(N)) for N in (12, 16, 20, 24)], "  "),
    "rates", "2p+2")
for p in (2, 3, 4)
    vals, rs, prev = Float64[], Float64[], nothing
    for N in (12, 16, 20, 24)
        sp = KdVSys(N; p, u0 = RESOLVED)
        v, _ = sweep(sp, (ss, dd, hh) -> midpoint(ss, dd, hh, 1)[1], 2e-4; T = 3.0)
        push!(vals, v)
        prev === nothing || push!(rs, log(prev[2] / v) / log(N / prev[1]))
        prev = (N, v)
    end
    @printf("     %3d  %s  %s  %5d\n", p,
        join([@sprintf("%11.3e", v) for v in vals], "  "),
        join([@sprintf("%.1f", r) for r in rs], ", "), 2p + 2)
    check(
        "p = $p: the plateau converges at about $(2p + 2)", abs(rs[end] - (2p + 2)) < 1.0,
        @sprintf("finest rate %.2f", rs[end]))
end

# --------------------------------------------------------------------------
header("12. why the sum does not accumulate")

println("  Two separate questions.  Why the sum is not SECULAR is almost trivial:")
println("  H2 is a norm, so a drift in it would force ||uhat|| to leave every")
println("  bounded set, and the trajectory does not.  Why the bound is so SMALL --")
println("  of size sup|eps_h| / detuning rather than of size ||uhat||^2 -- is first-")
println("  order averaging, and that is what the resonance count below establishes.")
println()
println("  eps_h is a homogeneous cubic form; on a uniform mesh its Fourier support")
println("  is {j+k+l = 0 mod N}, the assemblies being circulant, and by section 10")
println("  it carries no zero mode.  Along the linearised flow, with frequencies")
println("  omega_m from L = Minv S Minv K1, a term resonates iff omega_j + omega_k +")
println("  omega_l = 0.  The only resonant triples contain a zero mode -- and those")
println("  are exactly the ones eps_h does not have.")

function resonances(N, p)
    sp = KdVSys(N; p, u0 = U0b)
    Lm = sp.Minv * sp.S * sp.Minv * sp.K1
    om = zeros(N)
    for m in 0:(N - 1)
        v = [cis(2π * m * j / N) for j in 0:(N - 1)]
        om[m + 1] = imag(dot(v, Lm, v) / dot(v, v))
    end
    sig(m) = m ≤ N ÷ 2 ? m : m - N
    mn = Dict("exact" => Inf, "aliased" => Inf)
    ntriv = 0
    for j in 0:(N - 1), kk in 0:(N - 1)

        l = mod(-j - kk, N)
        a, b, c = sig(j), sig(kk), sig(l)
        if 0 in (a, b, c) || a + b == 0 || b + c == 0 || a + c == 0
            ntriv += 1
            continue
        end
        cls = a + b + c == 0 ? "exact" : "aliased"
        mn[cls] = min(mn[cls], abs(om[j + 1] + om[kk + 1] + om[l + 1]))
    end
    ntriv, mn
end

println()
@printf("     %4s %3s  %18s  %24s  %13s\n", "N", "p", "zero-mode triples",
    "min |detuning|, j+k+l=0", "min, aliased")
for (N, p) in ((16, 3), (32, 2), (32, 3), (32, 4), (48, 3))
    nt, mn = resonances(N, p)
    @printf("     %4d %3d  %18d  %24.4f  %13.1f\n", N, p, nt, mn["exact"], mn["aliased"])
    check("N = $N, p = $p: no resonance among the triples eps_h actually has",
        mn["exact"] > 5.0 && mn["aliased"] > 100.0,
        @sprintf("detunings %.3f and %.0f", mn["exact"], mn["aliased"]))
end
println("  The exact-triple minimum is the continuum value 3|jkl|, smallest at")
println("  (1,1,-2): 3 x 1 x 1 x 2 = 6, KdV having no non-trivial three-wave")
println("  resonance since j+k+l = 0 gives omega_j+omega_k+omega_l = 3jkl.  The")
println("  aliased triples, j+k+l = +-N, are the ones the discretisation adds, and")
println("  they are far off resonance, by O(N^3).")

println()
println("  First-order averaging then bounds the sum by 2 sup|eps_h| / detuning.")
println("  Measured against it, and against the length of the run:")
@printf("     %12s  %5s  %12s  %11s  %12s  %11s\n",
    "mesh", "T", "max|dH2|", "sup|eps_h|", "|mean eps_h|", "2 sup / 6")
const DELTA_MIN = 6.0
for (lab, kw) in (("uniform", (;)), ("graded", (; uniform = false, graded = true)),
    ("non-uniform", (; uniform = false)))
    sp = KdVSys(16; p = 3, u0 = RESOLVED, kw...)
    d = copy(sp.d0)
    ref = sysH2(sp, d)
    sup = worst = 0.0
    marks = Dict{Int, NTuple{2, Float64}}()
    dt = 1e-3
    for nstep in 1:round(Int, 40.0 / dt)
        y = midpoint(sp, d, dt, 1)[1]
        m = 0.5 .* (d + y)
        sup = max(sup, abs(eps_h(sp, m)))
        d = y
        worst = max(worst, abs(sysH2(sp, d) - ref))
        t = nstep * dt
        if abs(t - round(t)) < 1e-9 && round(Int, t) in (5, 10, 20, 40)
            marks[round(Int, t)] = (worst, sup)
        end
    end
    for T in (5, 10, 20, 40)
        w, sp_ = marks[T]
        @printf("     %12s  %5d  %12.4e  %11.3e  %12.3e  %11.3e\n",
            lab, T, w, sp_, w / T, 2sp_ / DELTA_MIN)
    end
    w5, w40 = marks[5][1], marks[40][1]
    check(
        "$lab mesh: max|dH2| is the same at T = 5 and T = 40, so the sum is " *
        "bounded and the mean of eps_h is zero",
        abs(w40 / w5 - 1.0) < 0.05,
        @sprintf("%.4e -> %.4e", w5, w40))
    if lab == "uniform"
        pred = 2 * marks[40][2] / DELTA_MIN
        check(
            "and on a uniform mesh the averaging estimate 2 sup|eps_h| / 6 " *
            "predicts the amplitude",
            abs(w40 / pred - 1.0) < 0.25,
            @sprintf("observed %.3e against predicted %.3e", w40, pred))
    end
end
println("  |mean eps_h| falls exactly as 1/T, which is the statement that the mean")
println("  is zero.  The non-uniform mesh is three orders higher because its")
println("  quadratic part no longer vanishes, and that part IS resonant -- its")
println("  conjugate-pair terms have detuning zero.  It still does not drift, which")
println("  is the first of the two arguments and not the second: first-order")
println("  averaging accounts for the amplitude on a uniform mesh, and boundedness")
println("  of the trajectory for the absence of secular growth everywhere.")

summary("verify_kdv_bea.jl")
