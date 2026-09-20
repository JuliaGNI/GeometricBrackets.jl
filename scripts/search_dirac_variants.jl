#!/usr/bin/env julia
#
# Search for a variant of the Dirac construction that restores the Jacobi identity.
#
#     julia --project=scripts scripts/search_dirac_variants.jl
#
# `verify_dirac_reduction.jl` establishes that the construction as stated does not work, and
# why: the Jacobiator transports tensorially through the Dirac projector,
#
#     [Jhat,Jhat]^ijk = P^i_l P^j_m P^k_n [J,J]^lmn ,        P = Pi_V1 R ,
#
# so Dirac reduction can only CREATE the Jacobi identity if [J,J] lands in ker P^{ox 3}, a
# subspace of dimension dim V2. That kernel is not empty, so the question is whether some
# variant of the construction aims into it. This searches, in order of prize:
#
#   B1  Repair the fine block. The coarse block c_ij^m, i,j in V1, is EXACT -- that is the
#       whole point of the hierarchical basis -- while [V1,V2] and [V2,V2] were truncated and
#       are therefore free. Is there a Lie algebra structure on V1 + V2 restricting to the
#       exact bracket on V1 x V1?
#         B1a  graded ansatz, exact.
#         B1b  the V1^3 equations alone, exact: they are LINEAR in [V2,V1].
#         B1c  ungraded, numerically, by Gauss-Newton onto the Lie-algebra variety.
#   B2  Deform the constraint surface: phi_a = u_a + sum_{i in V1} B_ai u_i, a linear shear.
#   B3  Choose V2 differently: every even-sized subset of a larger shell, with abelian
#       padding. Exhaustive and exact.
#   B4  Nested reduction. Ruled out by the composition lemma; see the verifier.
#
# Findings, in brief: B1a is flatly inconsistent from K=2 on, and its one solution at K=1 has
# [V2,V2] = 0, hence C = 0 and no Dirac bracket. B1b is consistent, so the obstruction is not
# in the coarse equations. B1c finds solutions at K=1 -- where V1 = sl(2) is already a
# subalgebra -- and none at K=2. B2 and B3 find nothing beyond the three-dimensional
# degeneracy at K=1.
#
# EXPLORATORY AND SLOW, and deliberately not part of run_all.jl. Only B1a needs SymPy; B1b
# and B3 are exact rational arithmetic and B1c and B2 are floating-point optimisation.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random
using SymPyPythonCall

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "rationals.jl"));
using .Rationals

const Q = Rational{BigInt}

"All strictly increasing k-tuples of `v`, as the Python's `itertools.combinations`."
function combos(v, k)
    out = Vector{eltype(v)}[]
    n = length(v)
    idx = collect(1:k)
    k > n && return out
    while true
        push!(out, [v[i] for i in idx])
        i = k
        while i ≥ 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && return out
        idx[i] += 1
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
end

# --------------------------------------------------------------------------
header("B1a. Graded Lie algebra structures on V1 + V2 (exact)")
# --------------------------------------------------------------------------

println("   Ansatz  [L_k,L_l] = g_kl L_{k+l},  g antisymmetric,  L_a = 0 for |a| > 2K,")
println("   with g_kl = l-k PINNED for |k|,|l| <= K.  Jacobi is then, for every triple")
println("   of distinct modes with |k+l+m| <= 2K,")
println()
println("       g_kl g_{k+l,m} + g_lm g_{l+m,k} + g_mk g_{m+k,l} = 0 .")
println()

function graded_system(K)
    M = 2K
    modes = collect((-M):M)
    inr(a) = abs(a) ≤ M
    pinned(k, l) = abs(k) ≤ K && abs(l) ≤ K
    sym = Dict{Tuple{Int, Int}, Sym}()
    unk = Sym[]

    function g(k, l)
        (!(inr(k) && inr(l)) || k == l) && return Sym(0)
        k > l && return -g(l, k)
        pinned(k, l) && return Sym(l - k)
        inr(k + l) || return Sym(0)
        if !haskey(sym, (k, l))
            sym[(k, l)] = sympy.Symbol("g[$k,$l]")
            push!(unk, sym[(k, l)])
        end
        sym[(k, l)]
    end

    eqs = Sym[]
    seen = Set{String}()
    for t in combos(modes, 3)
        k, l, m = t
        inr(k + l + m) || continue
        e = expand(g(k, l) * g(k + l, m) + g(l, m) * g(l + m, k) + g(m, k) * g(m + k, l))
        if e != 0 && !(string(e) in seen)
            push!(seen, string(e))
            push!(eqs, e)
        end
    end
    return modes, sym, unk, sort(eqs; by = string)
end

for K in (1, 2, 3)
    modes, sym, unk, eqs = graded_system(K)
    lin = isempty(unk) ? Sym[] :
          [e for e in eqs if sympy.Poly(e, unk...).total_degree() <= 1]
    sols = isempty(unk) ? [Dict{Sym, Sym}()] : solve(eqs, unk; dict = true)
    @printf("   K=%d: modes %d..%d, %d unknowns, %d equations (%d of them linear)\n",
        K, modes[1], modes[end], length(unk), length(eqs), length(lin))
    if !isempty(lin)
        inv_sym = Dict(v => kl for (kl, v) in sym)          # symbol -> (k,l)
        forced = solve(lin, unk; dict = true)
        if isempty(forced)
            println("        the LINEAR subsystem alone is already INCONSISTENT")
        else
            fixed = Dict(v => val for (v, val) in forced[1] if isempty(free_symbols(val)))
            witt = [v for (v, val) in fixed if val == Sym(inv_sym[v][2] - inv_sym[v][1])]
            @printf("        the linear subsystem determines %d unknown(s), %d of them at exactly the Witt value l-k\n",
                length(fixed), length(witt))
            if !isempty(witt)
                println("        e.g. " *
                        join(
                    ["$v = $(fixed[v])"
                     for v in sort(witt; by = string)[1:min(4, end)]], ", "))
            end
        end
    end
    @printf("        -> %d solution branch(es)\n", length(sols))
    for s0 in sols
        allzero = all(v == 0 for v in values(s0))
        println("           ",
            allzero ? "all mixed brackets vanish" :
            join(["$k => $v" for (k, v) in s0], ", "))
    end
    if K == 1
        check("K=1: the graded problem has solutions", length(sols) ≥ 1)
        check(
            "K=1: but every graded solution has [V2,V2] = 0, hence C = 0 and no " *
            "Dirac bracket",
            all(get(s0, sym[(-2, 2)], Sym(0)) == 0 for s0 in sols),
            "the only graded extension is sl(2) + abelian")
    else
        check("K=$K: the graded problem has NO solution at all", isempty(sols),
            "closure cannot be restored by any graded choice of the fine block")
    end
end

# --------------------------------------------------------------------------
header("B1b. The V1^3 equations alone are linear, and consistent (exact)")
# --------------------------------------------------------------------------

println("   For i,j,k in V1 the Jacobi condition involves c_ij^l (pinned) times")
println("   c_lk^m, which is pinned for l in V1 and unknown for l in V2.  It is")
println("   therefore LINEAR in the [V2,V1] block, and its consistency is decided by")
println("   comparing rank(A) with rank(A|b) over the rationals.")
println()

"Equation count, unknown count, rank(A) and rank(A|b) of the V1³ subsystem."
function stage1(C, keep, con)
    N = size(C, 1)
    # Column order is written out rather than taken from `Iterators.product`, which varies
    # the first index fastest where itertools varies the last.
    cols = Dict{Tuple{Int, Int, Int}, Int}()
    i = 0
    for a in con, b in keep, m in 1:N
        cols[(a, b, m)] = (i += 1)
    end
    ncol = i

    rows = Dict{Int, Q}[]
    rhs = Q[]
    for t in combos(keep, 3)
        ii, jj, kk = t
        for m in 1:N
            row = Dict{Int, Q}()
            const_ = zero(Q)
            function acc!(x, y, m, coef)
                iszero(coef) && return zero(Q)
                if x in keep && y in keep
                    return coef * C[m, x, y]
                elseif x in con && y in keep
                    q = cols[(x, y, m)]
                    row[q] = get(row, q, zero(Q)) + coef
                elseif x in keep && y in con
                    q = cols[(y, x, m)]
                    row[q] = get(row, q, zero(Q)) - coef
                else
                    error("stage 1 must not touch the [V2,V2] block")
                end
                return zero(Q)
            end
            for l in 1:N
                const_ += acc!(l, kk, m, C[l, ii, jj])
                const_ += acc!(l, ii, m, C[l, jj, kk])
                const_ += acc!(l, jj, m, C[l, kk, ii])
            end
            push!(rows, row)
            push!(rhs, -const_)
        end
    end
    A = zeros(Q, length(rows), ncol)
    for (r, row) in enumerate(rows), (q, v) in row

        A[r, q] = v
    end
    Ab = hcat(A, rhs)
    return size(A, 1), ncol, exact_rank(A), exact_rank(Ab)
end

println("        K   equations  unknowns  rank A  rank(A|b)   verdict")
for K in (1, 2, 3)
    tr = witt_truncation(K)
    ne, nc, rA, rAb = stage1(tr.C, tr.keep, tr.con)
    @printf("        %d   %7d   %7d   %5d   %7d    %s\n", K, ne, nc, rA, rAb,
        rA == rAb ? "consistent, sol. space dim $(nc - rA)" : "INCONSISTENT")
    check("K=$K: the V1^3 subsystem is consistent, so the obstruction is not there",
        rA == rAb)
end

# --------------------------------------------------------------------------
header("B1c. Ungraded Lie algebra structures, numerically")
# --------------------------------------------------------------------------

println("   Drop the grading: all c_ab^m with (a,b) not both in V1 are free.  Gauss-")
println("   Newton onto the Lie-algebra variety from random starts, then ask whether")
println("   any solution found has C = J22|Sigma invertible, i.e. is usable at all.")
println()

"The symmetrised bilinear form whose vanishing is the Lie-algebra condition."
function Bform(x, y)
    N = size(x, 1)
    R = zeros(N, N, N, N)
    @inbounds for n in 1:N, k in 1:N, j in 1:N, i in 1:N
        s = 0.0
        for m in 1:N
            s += x[m, i, j] * y[n, m, k] + x[m, j, k] * y[n, m, i] + x[m, k, i] * y[n, m, j]
        end
        R[n, i, j, k] = s
    end
    R
end

resnorm(cc) = norm(vec(Bform(cc, cc)))

function ungraded_search(K, ntrial; rseed = 0, itmax = 120)
    tr = witt_truncation(K)
    keep, con = tr.keep, tr.con
    N = size(tr.C, 1)
    c0 = Float64.(tr.C)
    pin = Set((i, j) for i in keep, j in keep)
    pairs = [(i, j) for i in 1:N for j in (i + 1):N if (i, j) ∉ pin]
    Es = [begin
              E = zeros(N, N, N)
              E[r, i, j] = 1.0
              E[r, j, i] = -1.0
              E
          end
          for (i, j) in pairs for r in 1:N]
    nx = length(Es)
    pack(x) = c0 + sum(x[t] .* Es[t] for t in 1:nx)

    "dR/dx as an (N^4, nx) matrix; R is quadratic, so dR[e] = B(e,c) + B(c,e)."
    jacmat(cc) = reduce(hcat, [vec(Bform(Es[t], cc) + Bform(cc, Es[t])) for t in 1:nx])

    rng = MersenneTwister(rseed)
    hits = Tuple{Float64, Float64}[]
    for _ in 1:ntrial
        x = 1.5 .* randn(rng, nx)
        for _ in 1:itmax
            cc = pack(x)
            r = vec(Bform(cc, cc))
            norm(r) < 1e-12 && break
            dx = jacmat(cc) \ (-r)
            s_, moved = 1.0, false
            for _ in 1:30
                if resnorm(pack(x .+ s_ .* dx)) < norm(r)
                    moved = true
                    break
                end
                s_ /= 2
            end
            moved || break
            x = x .+ s_ .* dx
        end
        cc = pack(x)
        rn = resnorm(cc)
        if rn < 1e-8
            u = zeros(N)
            u[keep] .= 1.0 .+ 3.0 .* rand(rng, length(keep))
            J = lie_poisson_matrix(cc, u)
            push!(hits, (rn, abs(det(J[con, con]))))
        end
    end
    return nx, hits
end

for (K, nt) in ((1, 25), (2, 25))
    nx, hits = ungraded_search(K, nt)
    best = isempty(hits) ? 0.0 : maximum(h[2] for h in hits)
    @printf("   K=%d: %d free coefficients, %d/%d starts reached the Lie-algebra variety\n",
        K, nx, length(hits), nt)
    flush(stdout)
    isempty(hits) || @printf("        best |det C| among them: %.4e\n", best)
    if K == 1
        check("K=1: ungraded Lie algebra extensions with C invertible DO exist",
            !isempty(hits) && best > 1e-6,
            "but V1 = sl(2) is already a subalgebra at K=1, so nothing is gained; " *
            "and the coarse block is 3-dimensional, hence degenerate")
    else
        check("K=$K: no ungraded Lie algebra extension found in $nt starts", isempty(hits),
            "consistent with B1a, which rules out the graded ones exactly")
    end
end

# --------------------------------------------------------------------------
header("B2. Deforming the constraint surface by a linear shear")
# --------------------------------------------------------------------------

println("   phi_a = u_a + sum_{i in V1} B_ai u_i is the change of coordinates v = S u")
println("   with S = [[I,0],[B,I]]; in v the constraints are coordinate constraints")
println("   again, so c^v_ij^n = S_ia S_jb c_ab^m (S^-1)_mn and the machinery applies.")
println("   Minimise the reduced Jacobiator over B.")
println()

"""
    reduced_diag(C, keep, con, u, tri)

Reduced Jacobiator, NORMALISED by max|Jhat|^2, plus its diagnostics.

Both the normalisation and the diagnostics are essential rather than cosmetic. Normalising is
needed because a shear driving Jhat towards zero would drive the raw Jacobiator to zero with
it; the diagnostics are needed because the OPPOSITE degeneracy is what actually happens here
-- the minimiser sends |B| to infinity, whereupon cond(J22) blows up and Jhat blows up with
it, so that dividing by max|Jhat|^2 makes the residual look small while the Dirac bracket is
in fact ceasing to exist.
"""
function reduced_diag(C, keep, con, u, tri)
    J = lie_poisson_matrix(C, u)
    Bi = inv(J[con, con])
    J12, J21 = J[keep, con], J[con, keep]
    Jh = J[keep, keep] - J12 * Bi * J21
    dJh = [C[l, keep, keep] - C[l, keep, con] * Bi * J21 +
           J12 * Bi * C[l, con, con] * Bi * J21 - J12 * Bi * C[l, con, keep] for l in keep]
    scale = maximum(abs, Jh)^2
    (isfinite(scale) && scale > 1e-300) || throw(SingularException(1))
    nk = length(keep)
    jac = [sum(dJh[l][i, j] * Jh[l, k] + dJh[l][j, k] * Jh[l, i] + dJh[l][k, i] * Jh[l, j]
           for l in 1:nk) for (i, j, k) in tri]
    return jac ./ scale, maximum(abs, Jh), cond(J[con, con])
end

"""
    shear_search(K; npts, ntrial, rseed, caps)

Minimise the normalised reduced Jacobiator over a linear shear of Sigma.

Reported as a sweep over a CAP on |B|, which is the decisive diagnostic and replaces any
single pass/fail threshold. If the best residual keeps falling as the cap is raised, while
|B| and cond(J22) rise to meet it, then the infimum sits on the boundary where the
constraints stop being second class and there is no interior solution -- which is what
happens here. A genuine solution would show the residual bottoming out at machine zero at
some finite cap and staying there.
"""
function shear_search(K; npts = 6, ntrial = 40, rseed = 0,
        caps = (1.0, 3.0, 10.0, 1e2, 1e3, Inf))
    tr = witt_truncation(K)
    keep, con = tr.keep, tr.con
    N = size(tr.C, 1)
    c0 = Float64.(tr.C)
    tri = [(t[1], t[2], t[3]) for t in combos(collect(1:length(keep)), 3)]
    rng = MersenneTwister(rseed)
    pts = [begin
               u = zeros(N)
               u[keep] .= 1.0 .+ 3.0 .* rand(rng, length(keep))
               u
           end
           for _ in 1:npts]
    idxs = [(a, i) for a in con for i in keep]
    nb = length(idxs)

    function transform(b)
        S = Matrix{Float64}(I, N, N)
        for (t, (a, i)) in enumerate(idxs)
            S[a, i] += b[t]
        end
        Sinv = inv(S)
        # Cv[n,i,j] = Σ_{a,b,m} S[i,a] S[j,b] C[m,a,b] Sinv[m,n], contracted a slot at a time
        T1 = zeros(N, N, N)                     # T1[n,a,b] = Σ_m Sinv[m,n] c0[m,a,b]
        @inbounds for n in 1:N, m in 1:N

            f = Sinv[m, n]
            iszero(f) && continue
            @views T1[n, :, :] .+= f .* c0[m, :, :]
        end
        T2 = zeros(N, N, N)                     # T2[n,i,b] = Σ_a S[i,a] T1[n,a,b]
        @inbounds for a in 1:N, i in 1:N

            f = S[i, a]
            iszero(f) && continue
            @views T2[:, i, :] .+= f .* T1[:, a, :]
        end
        Cv = zeros(N, N, N)                     # Cv[n,i,j] = Σ_b S[j,b] T2[n,i,b]
        @inbounds for b in 1:N, j in 1:N

            f = S[j, b]
            iszero(f) && continue
            @views Cv[:, :, j] .+= f .* T2[:, :, b]
        end
        Cv
    end

    function probe(b)
        cv = transform(b)
        rs = Float64[]
        mj = 0.0
        cn = 0.0
        for u in pts
            j, m, k = reduced_diag(cv, keep, con, u, tri)
            append!(rs, j)
            mj = max(mj, m)
            cn = max(cn, k)
        end
        rs, mj, cn
    end
    resid(b) = first(probe(b))
    base = norm(resid(zeros(nb)))

    # collect every iterate visited, then read off the best under each cap
    visited = NTuple{4, Float64}[]
    for _ in 1:ntrial
        b = 0.7 .* randn(rng, nb)
        for _ in 1:60
            local r, mj, cn
            try
                r, mj, cn = probe(b)
            catch
                break
            end
            nr = norm(r)
            push!(visited, (nr, norm(b), cn, mj))
            nr < 1e-13 && break
            Jm = Matrix{Float64}(undef, length(r), nb)
            e = zeros(nb)
            ok = true
            for t in 1:nb
                e[t] = 1e-7
                try
                    Jm[:, t] = (resid(b .+ e) .- resid(b .- e)) ./ 2e-7
                catch
                    ok = false
                    break
                end
                e[t] = 0.0
            end
            ok || break
            db = Jm \ (-r)
            s_, moved = 1.0, false
            for _ in 1:30
                try
                    if norm(resid(b .+ s_ .* db)) < nr
                        moved = true
                        break
                    end
                catch
                end
                s_ /= 2
            end
            moved || break
            b = b .+ s_ .* db
        end
    end

    rows = NTuple{5, Float64}[]
    for cap in caps
        adm = [v for v in visited if v[2] ≤ cap]
        if isempty(adm)
            push!(rows, (cap, base, 0.0, 1.0, 0.0))
            continue
        end
        v = argmin(first, adm)
        push!(rows, (cap, min(v[1], base), v[2], v[3], v[4]))
    end
    return nb, base, rows
end

for (K, nt) in ((1, 15), (2, 40))
    nb, base, rows = shear_search(K; ntrial = nt)
    @printf("   K=%d: %d shear unknowns, normalised |residual| at B=0 is %.4e\n", K, nb,
        base)
    println()
    println("        cap on |B|   best residual   |B| attained   cond(J22)   max|Jhat|")
    for (cap, r, bn, cn, mj) in rows
        cs = isfinite(cap) ? @sprintf("%6.0f", cap) : "  none"
        @printf("        %s       %.4e      %.3e     %.3e   %.3e\n", cs, r, bn, cn, mj)
    end
    println()
    if K == 1
        check("K=1: already zero without any shear -- the 3-dimensional degeneracy",
            base < 1e-9)
        continue
    end
    finite = [r for r in rows if isfinite(r[1])]
    check("K=$K: no shear at bounded |B| makes the reduced bracket Poisson",
        all(r -> r[2] > 1e-12, rows), "the residual never reaches machine zero at any cap")
    check(
        "K=$K: the residual keeps falling as the cap is raised, so the infimum " *
        "is on the boundary and there is no interior solution",
        finite[end][2] < 0.5 * finite[1][2] && rows[end][2] < finite[1][2],
        @sprintf("%.2e at |B|<=%.0f  ->  %.2e with no cap",
            finite[1][2], finite[1][1], rows[end][2]))
    check(
        "K=$K: and |B| and cond(J22) rise to meet each cap, i.e. the minimiser " *
        "is destroying second-classness rather than restoring Jacobi",
        rows[end][3] > 1e2 && rows[end][4] > 1e10,
        @sprintf("uncapped: |B| = %.2e, cond(J22) = %.2e, max|Jhat| = %.2e",
            rows[end][3], rows[end][4], rows[end][5]))
end

# --------------------------------------------------------------------------
header("B3. Choosing V2 differently: an exhaustive, exact sweep")
# --------------------------------------------------------------------------

println("   V1 = {|k| <= K} fixed; V2 ranges over every even-sized subset of the modes")
println("   with K < |k| <= M, for M = 2K .. 3K, optionally padded with abelian")
println("   directions.  A candidate counts as a hit only if the exact reduced")
println("   Jacobiator vanishes at three independent rational points.")
println()

"The truncated Witt algebra on |k| <= M, padded with `pad` abelian directions."
function witt_padded(K, M, pad)
    modes = collect((-M):M)
    pos = Dict(k => i for (i, k) in enumerate(modes))
    n = length(modes) + pad
    C = zeros(Q, n, n, n)
    for (a, k) in enumerate(modes), (b, l) in enumerate(modes)

        haskey(pos, k + l) && (C[pos[k + l], a, b] = Q(l - k))
    end
    keep = [pos[k] for k in modes if abs(k) ≤ K]
    rest = vcat([pos[k] for k in modes if abs(k) > K], collect((length(modes) + 1):n))
    return C, keep, rest, modes
end

for K in (1, 2)
    total = 0
    hits = Tuple{Int, Int, Int, Vector{String}}[]
    for M in (2K):(3K), pad in (0, 2)

        C, keep, rest, modes = witt_padded(K, M, pad)
        for size_ in 2:2:length(rest), con in combos(rest, size_)

            total += 1
            zero_ = true
            for sv in (101, 202, 303)
                u = zeros(Q, size(C, 1))
                rng = MersenneTwister(sv)
                for i in keep
                    u[i] = rnd(rng; lo = 1, hi = 9, dmax = 4)
                end
                try
                    Ĵ, dĴ = reduced_bracket(C, u, keep, con)
                    if !iszero(first(jacobi_residual(Ĵ, dĴ; normalised = false)))
                        zero_ = false
                        break
                    end
                catch
                    zero_ = false
                    break
                end
            end
            zero_ && push!(hits, (M, pad, size_,
                [i ≤ length(modes) ? string(modes[i]) : "z" for i in con]))
        end
    end
    @printf("   K=%d: %d candidate splittings tested, %d with a vanishing reduced Jacobiator\n",
        K, total, length(hits))
    for h in hits[1:min(5, end)]
        println("        M=$(h[1]) pad=$(h[2]) dim V2=$(h[3])  V2=[$(join(h[4], ", "))]")
    end
    length(hits) > 5 && println("        ... and $(length(hits) - 5) more")
    if K == 1
        check(
            "K=1: many splittings 'work', all of them with a 3-dimensional coarse " *
            "block",
            !isempty(hits),
            "every 3x3 antisymmetric bivector is decomposable, so Jacobi reduces " *
            "to Frobenius involutivity; this is not evidence")
    else
        check("K=$K: NO choice of V2 gives a Poisson reduced bracket", isempty(hits),
            "exhaustive over $total splittings")
    end
end

header("Conclusion")
println("""   No variant of the construction restores the Jacobi identity, and the
   verifier explains why in one sentence: closure by truncation and
   second-classness of the fine block exclude each other.

     * Closure requires a non-negative additive grading, which makes the high
       block an ideal.  Then Sigma is a Poisson submanifold, the PLAIN restriction
       is already Lie-Poisson -- and C = 0, so there is no Dirac bracket to form,
       and none is needed.
     * Whenever C is invertible the fine block is not an ideal, the truncation is
       not closed, and the projector identity forbids the reduction from repairing
       what the truncation broke.

   B2 deserves a footnote of its own.  Minimising over the shear has no minimiser:
   the infimum is approached only as |B| -> infinity, with cond(J22) and max|Jhat|
   diverging along the way, i.e. as the constraints cease to be second class.  The
   sequence is leaving the domain on which a Dirac bracket exists, not converging
   to a solution -- which is the singular hypersurface of the notes reappearing as
   the boundary of the admissible set.  Note also the two-sided trap: an
   unnormalised residual can be driven to zero by shrinking Jhat, and one
   normalised by max|Jhat|^2 by blowing Jhat up.  Report |B|, cond(J22) and
   max|Jhat| alongside the residual, and re-evaluate at points not used in the
   fit.

   What survives is in section 6 of verify_dirac_reduction.jl: the family of
   Theorem 2.1 is closed under Dirac reduction, acting by K -> Schur(K); and
   Dirac reduction of an honest Lie-Poisson bracket, the sine bracket in
   particular, gives an exactly Poisson but non-Lie-Poisson bracket on a coarse
   mode set.""")

summary("search_dirac_variants.jl")
