# `CollisionBracket` on a mapped domain: the physical frame, and the controls that break it.
#
# The bracket's gradients are physical. A space's derivative tables are the *parameter* ones,
# and the two agree only where the basis is written on the domain it is integrated over. On a
# mapped domain ∇_x = J⁻ᵀ∇̂, and a bracket that skips that factor is assembled in the wrong
# frame.
#
# **No structural check can see this.** Symmetry, positive semi-definiteness and the degeneracy
# (F,H) = 0 are algebraic properties of Q₂(z) = z⊥⊗z⊥ and of its annihilation of z; they say
# nothing about which z was handed in, and they hold in every parametrisation. Section 4 below
# measures that rather than asserting it, because a pass there is what makes the rest of this
# script necessary.
#
# Two checks follow, and they are not interchangeable:
#
#   1. **Covariance under a reparametrisation.** The same physical problem written twice must
#      give the same operator. This is the failure
#      `Experiments/MetriplecticRelaxation/scripts/verify_gradshafranov_disk.jl` §4 records,
#      and the numbers there are reproduced in Section 1.
#
#      It is **weaker than it looks, and Section 1 measures how**. A reparametrisation that
#      preserves a spline space is affine, so J is *constant*, and for a constant J the whole
#      bracket collapses to (det J⁻ᵀ)² times the parameter-frame one:
#
#          (B a)·((B c)⊥) = det(B) (a · c⊥)   for any 2×2 B,
#
#      and with one B at every node both contracted factors collapse that way. So a linear
#      family tests |det J| and the pairing, and **nothing about the direction** — a transposed
#      frame passes it exactly, which Section 1 shows rather than leaving as an assumption.
#
#   2. **An independent reference where J varies**, Section 2, which is the check the direction
#      has to pass. The annulus is the right case: the map is the polar chart, so the physical
#      gradient has a textbook closed form
#
#          ∂ₓu = cos θ ∂_r u − (sin θ / r) ∂_θ u ,   ∂_y u = sin θ ∂_r u + (cos θ / r) ∂_θ u ,
#
#      written out here by hand rather than taken from `PulledBack`, and the O(Nq²) double sum
#      of the bracket's own definition is assembled from it.
#
#      Section 3 runs the same reference on a `PolarSplineSpace` over the whole unit disk. That
#      is a separate statement and not a repetition: the polar index set is not a product, the
#      pairing is a sparse Cholesky rather than a Kronecker mass, and the three pole functions
#      reach around the entire angular axis.
#
# Run: julia --project=scripts --startup-file=no scripts/verify_frame_covariance.jl

using LinearAlgebra
using Printf
using Random
using PoissonBrackets
# By name rather than a bare `using`: `stiffness_matrix`, `weighted_matrix` and
# `derivative_matrix` are each a different generic in the two packages.
using SimpleSplines: UniformMesh, Dirichlet, Periodic, (..)

Random.seed!(20260918)

pass = Bool[]
record(name, ok) = (push!(pass, ok); @printf("  %-58s %s\n", name, ok ? "pass" : "FAIL"))
say(x) = (println(x); flush(stdout))
difference(A, B) = maximum(abs, A .- B) / maximum(abs, B)

## ---------------------------------------------------------------------------------------
## The linear family
## ---------------------------------------------------------------------------------------

# One fixed physical field, of the physical coordinates, so that it is the same function of
# position in every parametrisation.
physical(x) = sin(π * (x[1] - 1)) * sin(π * x[2]) * (1 + 0.3 * cos(3π * x[1]))

# F_a(x̂) = (x̂₁ + σ a x̂₂, a x̂₂) on [1,2] × [0, 1/a], so F_a = F_1 ∘ (x̂₁, a x̂₂): the physical
# domain, the physical spline basis and the coefficient vector are all independent of `a`.
# σ ≠ 0 makes J non-symmetric, and |det J| = a is constant, so the projection and the elliptic
# solve are unchanged to round-off and any residual difference is the bracket's.
function parametrisation(a; shear = 0.0, density = nothing, cells = 6, p = 2)
    s = TensorSplineSpace(
        (UniformMesh(cells, 1.0 .. 2.0), UniformMesh(cells, 0.0 .. 1.0 / a)),
        p, (Dirichlet(), Dirichlet()))
    F(x̂) = (x̂[1] + shear * a * x̂[2], a * x̂[2])
    DF(_) = [1.0 shear*a; 0.0 a]
    pb = PulledBack(s, F, DF; density)

    # Λ = (K^μ)⁻¹ M^μ, both assembled against the same pulled-back measure, so Λ is itself a
    # physical object and cannot be what a covariance failure is about.
    K = tensor_weighted_matrix(s, metric(pb))
    M = weighted_matrix(s, measure(pb), (0, 0), (0, 0))
    Λ = Matrix(cholesky(Symmetric(Matrix(K))) \ Matrix(M))
    û = project(s, [physical(x) for x in nodes(pb)])
    (; space = s, F, DF, pb, Λ, û, density,
        mass = dot(quadrature_weights(s) .* measure(pb), basis_values(s, (0, 0))' * û))
end

mobility(x, u) = 0.5 + 0.25 * x[1]^2 + 0.1 * u^2
mobility_derivative(x, u) = 0.2 * u

framed(P) = CollisionBracket(P.space, P.Λ, P.pb; mobility, mobility_derivative)

# CONTROL: the measure pulled back correctly and the frame not at all — the keyword form,
# which is right on an unmapped domain and is what every §5.4 run uses.
function frameless(P)
    CollisionBracket(P.space, P.Λ; density = measure(P.pb),
        mobility, mobility_derivative)
end

# The frame transposed. |det J| is unchanged by a transpose, so the measure, the pairing and
# every area are identical and only the direction is wrong.
function transposed(P)
    pbT = PulledBack(P.space, P.F, x -> Matrix(P.DF(x)'); density = P.density)
    CollisionBracket(P.space, P.Λ, pbT; mobility, mobility_derivative)
end

## ---------------------------------------------------------------------------------------
say("1. covariance under a linear reparametrisation")
say("")

# Case A is the setup of `Experiments/MetriplecticRelaxation/scripts/verify_gradshafranov_disk.jl`
# §4 exactly — no shear, unit density, unit mobility — so the control's fourth-power signature
# recorded there is reproduced here.
say("  CASE A   pure stretch, unit density, unit mobility")
let
    base = parametrisation(1.0)
    Gf = metric_matrix(CollisionBracket(base.space, base.Λ; density = measure(base.pb)),
        base.û)
    Gp = metric_matrix(CollisionBracket(base.space, base.Λ, base.pb), base.û)
    @printf("    %-6s %-14s %-12s %-14s %s\n",
        "scale", "∫u dx", "‖G‖ framed", "rel to scale 1", "‖G‖ CONTROL / scale 1")
    ok_same = true
    ok_framed = true
    ratios = Float64[]
    for a in (1.0, 2.0, 4.0)
        P = parametrisation(a)
        G = metric_matrix(CollisionBracket(P.space, P.Λ, P.pb), P.û)
        C = metric_matrix(CollisionBracket(P.space, P.Λ; density = measure(P.pb)), P.û)
        push!(ratios, maximum(abs, C) / maximum(abs, Gf))
        @printf("    %4.1f   %.10f   %.4e   %.3e      %8.2f\n",
            a, P.mass, maximum(abs, G), difference(G, Gp), ratios[end])
        ok_same &= abs(P.mass - base.mass) < 1e-12
        ok_framed &= difference(G, Gp) < 1e-11
    end
    record("the three are one physical problem (∫u dx identical)", ok_same)
    record("the framed bracket is invariant", ok_framed)
    record("CONTROL the frameless bracket scales as the fourth power",
        abs(ratios[2] - 16) < 1 && abs(ratios[3] / ratios[2] - 16) < 1)
end
say("")

say("  CASE B   shear σ = 0.7, density 1/r, state-dependent mobility")
let
    opts = (; shear = 0.7, density = x -> 1 / x[1])
    base = parametrisation(1.0; opts...)
    Gp = metric_matrix(framed(base), base.û)
    Gf = metric_matrix(frameless(base), base.û)
    Gt = metric_matrix(transposed(base), base.û)

    @printf("    %-6s %-14s %-12s %-12s %s\n",
        "scale", "∫u dμ", "framed", "CONTROL none", "transposed")
    ok_same = true
    ok_framed = true
    for a in (1.0, 2.0, 4.0)
        P = parametrisation(a; opts...)
        rp = difference(metric_matrix(framed(P), P.û), Gp)
        @printf("    %4.1f   %.10f   %.3e    %.3e    %.3e\n", a, P.mass, rp,
            difference(metric_matrix(frameless(P), P.û), Gf),
            difference(metric_matrix(transposed(P), P.û), Gt))
        ok_same &= abs(P.mass - base.mass) < 1e-12
        ok_framed &= rp < 1e-10
    end
    record("the three are one physical problem (∫u dμ identical)", ok_same)
    record("the framed bracket is invariant", ok_framed)
    record("CONTROL dropping the frame is an O(1) different operator",
        difference(Gf, Gp) > 0.1)

    # NOT a control, and this is the point of measuring it. For a constant J the bracket
    # depends on the frame only through det J, which a transpose leaves alone, so the
    # transposed frame reproduces the correct operator here exactly. A check that cannot fail
    # establishes nothing, and this one would have been read as a pass.
    rt = difference(Gt, Gp)
    say(@sprintf("    frameless vs framed %.3e   transposed vs framed %.3e",
        difference(Gf, Gp), rt))
    record("a transposed frame is INVISIBLE to a linear family, as predicted", rt < 1e-13)
    record("the transposed Jacobian is nonetheless caught by jacobian_residual",
        jacobian_residual(base.F, x -> Matrix(base.DF(x)'),
            quadrature_nodes(base.space)) > 0.1)
end
say("")
say("""  So Section 1 is necessary and not sufficient: it pins |det J| and the pairing, and
  says nothing about the direction the gradient is read in.""")
say("")

## ---------------------------------------------------------------------------------------
say("2. the annulus, where J varies: against an independent reference")
say("")

const R₀, R₁ = 0.4, 1.3
Fpolar(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
DFpolar(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]

# The bracket's own definition, summed over pairs of quadrature nodes, with the physical
# gradient written out by hand from the polar chart rather than read off `PulledBack`:
#
#   (F,G) = ½ ∫∫ M M′ (∇f − ∇f′)·z⊥  z⊥·(∇g − ∇g′) dμ′ dμ ,   z = ∇φ − ∇φ′ ,
#
# which in the basis is ½ ΣΣ c c′ v v′ᵀ with v_K = (∇Φ_K − ∇Φ′_K)·z⊥.
function polar_reference(s, φ̂, û, mob)
    x̂ = quadrature_nodes(s)
    w = quadrature_weights(s)
    r = [pt[1] for pt in x̂]
    θ = [pt[2] for pt in x̂]
    P̂r = Matrix(basis_values(s, (1, 0)))
    P̂θ = Matrix(basis_values(s, (0, 1)))

    # ∂ₓ = cos θ ∂_r − (sin θ / r) ∂_θ,  ∂_y = sin θ ∂_r + (cos θ / r) ∂_θ
    Px = P̂r .* cos.(θ)' .- P̂θ .* (sin.(θ) ./ r)'
    Py = P̂r .* sin.(θ)' .+ P̂θ .* (cos.(θ) ./ r)'

    u = Matrix(basis_values(s, (0, 0)))' * û
    xphys = [Fpolar(pt) for pt in x̂]
    # |det J| = r for the polar chart, by hand as well.
    c = [w[q] * r[q] * mob(xphys[q], u[q]) for q in eachindex(w)]

    gφ = (Px' * φ̂, Py' * φ̂)
    N, Nq = size(Px, 1), length(w)
    A = zeros(N, N)
    for q in 1:Nq, q′ in 1:Nq

        z = (gφ[1][q] - gφ[1][q′], gφ[2][q] - gφ[2][q′])
        v = (-z[2]) .* (Px[:, q] .- Px[:, q′]) .+ z[1] .* (Py[:, q] .- Py[:, q′])
        A .+= (0.5 * c[q] * c[q′]) .* (v * v')
    end
    return A
end

let
    s = TensorSplineSpace((UniformMesh(4, R₀ .. R₁), UniformMesh(8, 0 .. 2π)),
        2, (Dirichlet(), Periodic()))
    pb = PulledBack(s, Fpolar, DFpolar)
    N = nbasis(s)
    φ̂ = randn(N)
    û = randn(N)

    mob(x, u) = 0.7 + 0.3 * x[1]^2 + 0.05 * u^2
    dmob(x, u) = 0.1 * u

    b = CollisionBracket(s, φ̂, pb; mobility = mob, mobility_derivative = dmob)
    A = metric_operator(b, û)
    Aref = polar_reference(s, φ̂, û, mob)
    @printf("    %d basis functions, %d quadrature nodes\n", N,
        length(quadrature_weights(s)))
    @printf("    metric_operator against the O(Nq²) reference   %.3e\n",
        difference(A, Aref))
    record("the framed bracket is the operator its definition names",
        difference(A, Aref) < 1e-10)

    # CONTROL: the frame dropped and nothing else. The mobility is composed with the map here,
    # so the kernel weights c are identical to the framed bracket's and the *only* difference
    # is the frame.
    bf = CollisionBracket(s, φ̂; density = measure(pb),
        mobility = (p, u) -> mob(Fpolar(p), u), mobility_derivative = dmob)
    Af = metric_operator(bf, û)
    @printf("    CONTROL frame dropped                          %.3e\n",
        difference(Af, Aref))
    record("CONTROL dropping the frame is an O(1) different operator",
        difference(Af, Aref) > 0.1)

    # CONTROL: the frame transposed. Invisible in Section 1; here J varies and it is not.
    pbT = PulledBack(s, Fpolar, x -> Matrix(DFpolar(x)'))
    bt = CollisionBracket(s, φ̂, pbT; mobility = mob, mobility_derivative = dmob)
    @printf("    CONTROL frame transposed                       %.3e\n",
        difference(metric_operator(bt, û), Aref))
    record("CONTROL transposing the frame is an O(1) different operator",
        difference(metric_operator(bt, û), Aref) > 0.1)

    # The mobility is sampled at the physical points, not the parameter ones. The reference
    # above already depends on this, so it is stated separately rather than left implicit.
    bx = CollisionBracket(s, φ̂, pb; mobility = (x, u) -> x[1], mobility_derivative = 0)
    sampled = PoissonBrackets._collision_state(bx, û).M
    @printf("    mobility sampled at F(x̂), not at x̂             %.3e   (parameter %.3e)\n",
        maximum(abs, sampled .- [x[1] for x in nodes(pb)]),
        maximum(abs, sampled .- [x[1] for x in quadrature_nodes(s)]))
    record("a mobility is a function of the physical point",
        maximum(abs, sampled .- [x[1] for x in nodes(pb)]) < 1e-14 &&
            maximum(abs, sampled .- [x[1] for x in quadrature_nodes(s)]) > 0.1)
end
say("")

## ---------------------------------------------------------------------------------------
say("3. the polar space, across the pole")
say("")

# The space the whole polar branch exists for. Its index set is not a product, its mass
# factorisation is a sparse Cholesky rather than a Kronecker one, and the three pole functions
# reach around the entire angular axis — so that the frame reaches it at all is a separate
# statement from Section 2, and the same hand-written reference settles it. The chart is the
# same polar one, on the whole unit disk this time: 1/s is singular at the pole and integrable,
# and no quadrature node sits there.
let
    s = PolarSplineSpace((4, 8), 2)
    pb = PulledBack(s, Fpolar, DFpolar)
    N = nbasis(s)
    φ̂, û = randn(N), randn(N)
    mob(x, u) = 0.7 + 0.3 * x[1]^2 + 0.05 * u^2
    dmob(x, u) = 0.1 * u

    b = CollisionBracket(s, φ̂, pb; mobility = mob, mobility_derivative = dmob)
    Aref = polar_reference(s, φ̂, û, mob)
    @printf("    %d basis functions (3 of them pole functions), %d quadrature nodes\n",
        N, length(quadrature_weights(s)))
    @printf("    metric_operator against the O(Nq²) reference   %.3e\n",
        difference(metric_operator(b, û), Aref))
    record("the frame reaches the polar space too",
        difference(metric_operator(b, û), Aref) < 1e-10)

    bf = CollisionBracket(s, φ̂; density = measure(pb),
        mobility = (p, u) -> mob(Fpolar(p), u), mobility_derivative = dmob)
    @printf("    CONTROL frame dropped                          %.3e\n",
        difference(metric_operator(bf, û), Aref))
    record("CONTROL dropping the frame is an O(1) different operator",
        difference(metric_operator(bf, û), Aref) > 0.1)

    record("symmetry, semidefiniteness and degeneracy hold on the polar space",
        issymmetric(b, û) && ispositive_semidefinite(b, û) &&
            degeneracy_residual(b, û) < 1e-11)
    @printf("    symmetric %s   semidefinite %s   (F,H) = %.2e\n",
        issymmetric(b, û), ispositive_semidefinite(b, û), degeneracy_residual(b, û))
end
say("")

## ---------------------------------------------------------------------------------------
say("4. the structural checks pass in every frame, which is why they are not the check")
say("")

let
    opts = (; shear = 0.7, density = x -> 1 / x[1])
    for a in (1.0, 4.0),
        (name, build) in (("framed", framed), ("frameless", frameless),
            ("transposed", transposed))

        P = parametrisation(a; opts...)
        b = build(P)
        sym = issymmetric(b, P.û)
        psd = ispositive_semidefinite(b, P.û)
        deg = degeneracy_residual(b, P.û)
        @printf("    scale %.1f  %-11s  symmetric %-5s  semidefinite %-5s  (F,H) = %.2e\n",
            a, name, sym, psd, deg)
        record("scale $a, $name: symmetry, semidefiniteness and degeneracy hold",
            sym && psd && deg < 1e-11)
    end
end
say("")
say("""  All six pass. A bracket in the wrong frame is symmetric, semidefinite and degenerate,
  so `verify_polar_bracket.jl`'s structural pass is necessary and not sufficient on a mapped
  domain — and the degeneracy passing is not a free result either: it holds only because the
  pairing of the sandwich moved with the frame.""")
say("")

## ---------------------------------------------------------------------------------------
say("5. the unmapped path is unchanged")
say("")

# An identity map has J = I, so a bracket given its pullback must agree with the bracket given
# the same density and no pullback — to round-off rather than bit-for-bit, the two pairings
# being a Kronecker mass operator and a sparse Cholesky of the same matrix.
let
    s = TensorSplineSpace((UniformMesh(6, 1.0 .. 2.0), UniformMesh(6, 0.0 .. 1.0)),
        2, (Dirichlet(), Dirichlet()))
    ρ(x) = 1 / x[1]
    pb = PulledBack(s, x -> (x[1], x[2]), _ -> [1.0 0.0; 0.0 1.0]; density = ρ)
    K = tensor_weighted_matrix(s, metric(pb))
    M = weighted_matrix(s, measure(pb), (0, 0), (0, 0))
    Λ = Matrix(cholesky(Symmetric(Matrix(K))) \ Matrix(M))
    û = project(s, [physical(x) for x in quadrature_nodes(s)])

    G1 = metric_matrix(
        CollisionBracket(s, Λ; density = ρ, mobility, mobility_derivative), û)
    G2 = metric_matrix(CollisionBracket(s, Λ, pb; mobility, mobility_derivative), û)
    @printf("    identity map, relative difference   %.3e\n", difference(G2, G1))
    record("an identity pullback reproduces the plain bracket", difference(G2, G1) < 1e-12)

    # `volume_element` is 1 there, and the mass it builds is the space's own.
    𝕄 = weighted_matrix(s, volume_element(pb), (0, 0), (0, 0))
    @printf("    ‖∫ΦΦ|det J| − mass_matrix‖          %.3e\n",
        maximum(abs, Matrix(𝕄) .- Matrix(mass_matrix(s))))
    record("the mapped pairing reduces to the space's mass matrix",
        maximum(abs, Matrix(𝕄) .- Matrix(mass_matrix(s))) < 1e-14)
end
say("")

## ---------------------------------------------------------------------------------------
say("6. the routes through the operator agree on a mapped domain")
say("")

# `metric_apply` goes through the fourteen scalar moments and `metric_matrix` through the
# R,S,T factorisation; they are independent evaluations of the same operator, and the frame
# enters each separately. `metric_directional` and `metric_derivative` are the two spellings
# of the state derivative, and both read the tables again.
let
    P = parametrisation(2.0; shear = 0.7, density = x -> 1 / x[1])
    b = framed(P)
    N = nbasis(P.space)
    G = metric_matrix(b, P.û)
    c = randn(N)

    rapply = norm(metric_apply(b, P.û, c) .- G * c) / norm(G * c)
    @printf("    metric_apply against metric_matrix * c        %.3e\n", rapply)
    record("the moment route and the assembly agree", rapply < 1e-10)

    v = randn(N)
    D = metric_directional(b, P.û, v)
    dG = metric_derivative(b, P.û)
    rdir = maximum(abs, D .- [dot(dG[m, i, :], v) for i in 1:N, m in 1:N]) / maximum(abs, D)
    @printf("    metric_directional against metric_derivative  %.3e\n", rdir)
    record("the contracted tensor and the directional derivative agree", rdir < 1e-10)

    # A central difference of the assembled bracket, which reads no derivative code at all.
    ε = 1e-6
    m = 7
    e = zeros(N)
    e[m] = 1
    fd = (metric_matrix(b, P.û .+ ε .* e) * v .- metric_matrix(b, P.û .- ε .* e) * v) ./
         (2ε)
    rfd = norm(D[:, m] .- fd) / norm(fd)
    @printf("    analytic derivative against a difference      %.3e\n", rfd)
    record("the analytic state derivative is the right one", rfd < 1e-6)
end
say("")

## ---------------------------------------------------------------------------------------
say(all(pass) ? "ALL CHECKS PASS" :
    "SOME CHECK FAILED ($(count(!, pass)) of $(length(pass)))")
exit(all(pass) ? 0 : 1)
