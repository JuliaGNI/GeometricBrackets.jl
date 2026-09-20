# The mapped Laplacian and the collision bracket on a `PolarSplineSpace`, pole included.
#
# Two things are checked, and the first is the one a tensor-product space cannot be asked at
# all, because it is not even C⁰ at the pole:
#
#   1. the pulled-back stiffness on the *whole* unit disk is the Laplacian. `u = 1 − s²` is in
#      the polar space **exactly** — it is radial, so its θ-dependence at the pole is constant
#      and its second row of coefficients is too — it vanishes exactly on the rim, and
#      ∫|∇u|² dx = 2π in closed form. So the assembled number has an exact target and not a
#      refined one.
#
#   2. the `CollisionBracket`'s structural properties survive the space change: symmetry,
#      positive semi-definiteness, and the degeneracy (F,H) = 0 — each against a control that
#      must fail. A space change can break the degeneracy without breaking symmetry, which is
#      why the third is measured separately and not inferred from the first two.
#
# Both are then repeated against the measure of `eq:mapping`, which is the configuration the
# Grad-Shafranov case C2 actually runs in.
#
# Run: julia --project=scripts --startup-file=no scripts/verify_polar_bracket.jl

using LinearAlgebra
using Printf
using Random
using GeometricBrackets
using SimpleSplines: (..)

Random.seed!(20260917)

pass = Bool[]
record(name, ok) = (push!(pass, ok); @printf("  %-58s %s\n", name, ok ? "pass" : "FAIL"))

## ---------------------------------------------------------------------------------------
## 1. The mapped Laplacian on the whole disk
## ---------------------------------------------------------------------------------------

# The polar chart of the unit disk. det J = s, which vanishes at the pole, so the metric
# coefficient m J⁻¹J⁻ᵀ = diag(s, 1/s) is singular there. That is the true polar Laplacian and
# not an artefact: for a function that is C¹ at the pole, ∂_θu = O(s), so (1/s)(∂_θu)² is
# integrable. No quadrature node sits at s = 0 — Gauss nodes are interior to their cell — so
# the coefficient is large but finite, and the C¹ property of the space is what makes the
# assembled integral converge rather than merely exist.
Fdisk(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
DFdisk(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]

println("1. the pulled-back Laplacian on the whole unit disk, pole included")
println()

for cells in ((16, 32), (32, 64))
    s = PolarSplineSpace(cells, 3)
    pb = PulledBack(s, Fdisk, DFdisk)
    K = tensor_weighted_matrix(s, metric(pb))

    # u = 1 − s², exactly in the space: radial, so every row of coefficients is constant in θ,
    # which satisfies the pole constraints with zero linear part. Projected rather than
    # written out by hand, and the projection error is then reported, which is the check that
    # it really is exact.
    û = project(s, x -> 1 - x[1]^2)
    proj_err = maximum(abs(evaluate(s, û, x) - (1 - x[1]^2))
    for x in [(0.0, 0.3), (0.0, 2.7), (0.25, 1.0), (0.5, 4.0), (1.0, 0.0)])

    energy = dot(û, K * û)
    @printf("  %2d × %-3d  N = %5d   ∫|∇u|² dx = %.10f   exact 2π = %.10f   rel %.3e\n",
        cells[1], cells[2], nbasis(s), energy, 2π, abs(energy - 2π) / 2π)
    @printf("            u = 1 − s² is in the space: max projection error %.3e\n", proj_err)

    if cells == (32, 64)
        record("energy against the closed form", abs(energy - 2π) / 2π < 1e-10)
        record("u = 1 − s² exactly in the space", proj_err < 1e-11)

        # CONTROL: the plain parameter-square stiffness, no pullback. A different operator.
        plain = dot(û, stiffness_matrix(s) * û)
        @printf("  CONTROL plain stiffness, no pullback = %.8f   rel %.3f\n",
            plain, abs(plain - 2π) / 2π)
        record("CONTROL plain stiffness is not the Laplacian", abs(plain - 2π) / 2π > 1e-2)

        # CONTROL: measure kept, metric dropped to |det J| times the identity — the 1/s of the
        # θθ component forgotten.
        #
        # On `u = 1 − s²` this control is *invisible*, and that is worth stating rather than
        # arranging around: `u` is radial, so ∂_θu = 0 and the θθ component never appears. The
        # true metric is diag(s, 1/s) and the dropped one diag(s, s); they agree exactly on
        # every radial field. **A radial test field cannot detect a dropped angular metric.**
        # So the control is run against a field with angular structure, which vanishes to
        # second order at the pole and is therefore safely in the space.
        𝔹 = [k == l ? copy(measure(pb)) : zeros(length(measure(pb))) for k in 1:2, l in 1:2]
        Kbad = tensor_weighted_matrix(s, 𝔹)

        drop_radial = dot(û, Kbad * û)
        @printf("  CONTROL dropped metric, on the RADIAL u  = %.8f   rel %.3e  (invisible)\n",
            drop_radial, abs(drop_radial - 2π) / 2π)
        record("a radial field cannot see a dropped angular metric",
            abs(drop_radial - 2π) / 2π < 1e-10)

        ŵ = project(s, x -> (1 - x[1]^2) * x[1]^2 * cos(2 * x[2]))
        true_w = dot(ŵ, K * ŵ)
        drop_w = dot(ŵ, Kbad * ŵ)
        @printf("  CONTROL dropped metric, with ANGULAR structure: %.8f against %.8f   rel %.3f\n",
            drop_w, true_w, abs(drop_w - true_w) / abs(true_w))
        record("CONTROL dropped metric is not the Laplacian",
            abs(drop_w - true_w) / abs(true_w) > 1e-2)

        # The area of the disk, for scale: the measure alone is right even when the metric is
        # not, which is exactly why the second control is needed.
        @printf("  for scale: ∫1 dx = %.12f against π = %.12f\n",
            dot(quadrature_weights(s), measure(pb)), π)
        record("the measure alone gives the disk's area",
            abs(dot(quadrature_weights(s), measure(pb)) - π) / π < 1e-12)
    end
end
println()

## ---------------------------------------------------------------------------------------
## 2. The collision bracket on the polar space
## ---------------------------------------------------------------------------------------

# `eq:mapping`, the Grad-Shafranov case C2 geometry, and its measure dμ = dr dz / r.
const MAPC = (e = 1.4, ε = 0.3, a = 4.0, b = 3.0, c = 6.3, ξ = 1 / sqrt(1 - 0.3^2 / 4))

function gs_map(x)
    (; e, ε, a, b, c, ξ) = MAPC
    s, θ = x[1], x[2]
    q = sqrt(1 + ε * (ε + 2s * cos(θ)))
    (a * (b + (1 - q) / ε), c * e * ξ * s * sin(θ) / (2 - q))
end

function gs_jacobian(x)
    (; e, ε, a, b, c, ξ) = MAPC
    s, θ = x[1], x[2]
    q = sqrt(1 + ε * (ε + 2s * cos(θ)))
    qs = ε * cos(θ) / q
    qθ = -ε * s * sin(θ) / q
    k = c * e * ξ
    [-a*qs/ε -a*qθ/ε;
     k*(sin(θ) * (2 - q) + s * sin(θ) * qs)/(2 - q)^2 k*(s * cos(θ) * (2 - q) + s * sin(θ) * qθ)/(2 - q)^2]
end

function structural(label, s, density)
    N = nbasis(s)
    M = Matrix(mass_matrix(s))
    # A generating map with the same shape the Grad-Shafranov flow uses: Λ = (K + εM)⁻¹M, so
    # the Hamiltonian is in the bracket's kernel by construction.
    Λ = (Matrix(stiffness_matrix(s)) + 0.5M) \ M
    b = CollisionBracket(s, Λ; density = density)
    û = randn(N)

    G = metric_matrix(b, û)
    λ = eigvals(Symmetric(G))
    sym = maximum(abs, G - G') / maximum(abs, G)
    deg = degeneracy_residual(b, û)
    degc = degeneracy_residual(b, û, randn(N))

    println("  ", label, ", N = ", N)
    @printf("    symmetry  ‖G−Gᵀ‖/‖G‖ = %.3e      semidefinite  λmin/λmax = %.3e\n",
        sym, minimum(λ) / maximum(λ))
    @printf("    degeneracy (F,H)     = %.3e      CONTROL wrong generator = %.3e\n",
        deg, degc)
    record("$(label): symmetric", sym < 1e-13)
    record("$(label): positive semi-definite", minimum(λ) > -1e-11 * maximum(λ))
    record("$(label): singular, not definite", minimum(λ) < 1e-11 * maximum(λ))
    record("$(label): degenerate on its Hamiltonian", deg < 1e-12)
    record("$(label): CONTROL a wrong generator is not", degc > 0.1)
    return (s = s, b = b, û = û, Λ = Λ)
end

println("2. the collision bracket's structural properties on the polar space")
println()

sp = PolarSplineSpace((6, 12), 2)
flat = structural("parameter measure", sp, one(Float64))

pbgs = PulledBack(sp, gs_map, gs_jacobian; density = x -> 1 / x[1])
@printf("  eq:mapping Jacobian against a central difference: %.3e\n",
    jacobian_residual(gs_map, gs_jacobian, quadrature_nodes(sp)))
mapped = structural("eq:mapping, dμ = dr dz / r", sp, measure(pbgs))
println()

## ---------------------------------------------------------------------------------------
## 3. The controls that must break positivity
## ---------------------------------------------------------------------------------------

println("3. controls that must break positivity or degeneracy")
println()

# A mobility that changes sign. `eq:M-condition` requires M > 0, and nothing enforces it; a
# sign-changing M makes κ an indefinite kernel and the bracket is no longer a Gram matrix.
bsign = CollisionBracket(sp, flat.Λ; mobility = (x, u) -> cos(6 * x[2]),
    mobility_derivative = 0)
λsign = eigvals(Symmetric(metric_matrix(bsign, flat.û)))
@printf("  sign-changing mobility      λmin/λmax = %+.3e  (must be clearly negative)\n",
    minimum(λsign) / maximum(λsign))
record("CONTROL a sign-changing mobility breaks positivity",
    minimum(λsign) < -1e-6 * maximum(λsign))

# The perpendicular dropped: z ⊗ z in place of z⊥ ⊗ z⊥. A Gram matrix either way, so it stays
# symmetric and positive semi-definite — only the degeneracy notices. The O(Nq²) reference
# form, which is why the mesh here is small.
function double_sum(b, û; kernel)
    s = b.space
    st = GeometricBrackets._collision_state(b, û)
    P = (Matrix(basis_values(s, (1, 0))), Matrix(basis_values(s, (0, 1))))
    # `st.c = M μ` is the weight. `st.γ` is the recentred **perped** gradient, so the
    # *unperped* one is (γ₂, −γ₁); recentring subtracts a constant and only differences appear
    # below, which is the freedom of origin the bracket is built on.
    #
    # The unperped gradient is what must be fed here. Handing the perped one to `z -> z*z'`
    # reconstructs z⊥ ⊗ z⊥, which is the *correct* kernel — a control that cannot fail.
    c = st.c
    g = (st.γ[2], -st.γ[1])
    N, Nq = nbasis(s), length(c)
    A = zeros(N, N)
    for r in 1:Nq, r′ in 1:Nq

        K = kernel([g[1][r] - g[1][r′], g[2][r] - g[2][r′]])
        D = (P[1][:, r] .- P[1][:, r′], P[2][:, r] .- P[2][:, r′])
        f = 0.5 * c[r] * c[r′]
        for l in 1:2, k in 1:2

            A .+= (f * K[k, l]) .* (D[k] * D[l]')
        end
    end
    return A
end

ssmall = PolarSplineSpace((4, 8), 2)
Msm = Matrix(mass_matrix(ssmall))
Λsm = (Matrix(stiffness_matrix(ssmall)) + 0.5Msm) \ Msm
bsm = CollisionBracket(ssmall, Λsm)
ûsm = randn(nbasis(ssmall))
φ̂sm = Λsm * ûsm

perp(z) = [-z[2], z[1]]
degen(A) = maximum(abs, A * φ̂sm) / (maximum(abs, A) * maximum(abs, φ̂sm))

# The correct kernel first, which also checks this O(Nq²) reference against the package's own
# collapsed form: if the reference were wrong, the control below would prove nothing.
Aok = double_sum(bsm, ûsm; kernel = z -> (p = perp(z); p * p'))
@printf("  reference with z⊥⊗z⊥       symmetry %.3e   λmin/λmax %+.3e   (F,H) %.3e\n",
    maximum(abs, Aok - Aok') / maximum(abs, Aok),
    minimum(eigvals(Symmetric(Aok))) / maximum(eigvals(Symmetric(Aok))), degen(Aok))
record("the O(Nq²) reference is degenerate, as the collapsed form is", degen(Aok) < 1e-10)

Anp = double_sum(bsm, ûsm; kernel = z -> z * z')
λnp = eigvals(Symmetric(Anp))
resid = degen(Anp)
@printf("  CONTROL z⊗z, the ⊥ dropped  symmetry %.3e   λmin/λmax %+.3e   (F,H) %.3e\n",
    maximum(abs, Anp - Anp') / maximum(abs, Anp), minimum(λnp) / maximum(λnp), resid)
record("CONTROL z⊗z stays symmetric", maximum(abs, Anp - Anp') / maximum(abs, Anp) < 1e-10)
record("CONTROL z⊗z stays positive semi-definite", minimum(λnp) > -1e-9 * maximum(λnp))
record("CONTROL z⊗z breaks the degeneracy", resid > 0.01)
println()

## ---------------------------------------------------------------------------------------
## 4. The recentring regime
## ---------------------------------------------------------------------------------------

# A relaxed Grad-Shafranov state is a large mean ∇φ with a small variation, which is the regime
# where the uncentred moment form loses its significant digits and 𝔻_s goes indefinite. The
# centring is in `_diffusion_tensor` and is space-independent — `scripts/verify_metric_collapse.jl`
# establishes that with no grid, basis or space at all — so what is checked here is that the
# polar space reaches the same conclusion with its own weights, at the spread that breaks the
# uncentred form.
println("4. the recentring regime: large mean gradient, small variation")
println()

b = CollisionBracket(sp, flat.Λ; density = measure(pbgs))

for spread in (1e-3, 1e-5, 1e-7)
    # A state whose generating field has a large mean gradient and a variation of `spread`.
    ĥ = flat.Λ \ project(sp, x -> 12 * x[1] + spread * sin(5 * x[2]))
    λ = eigvals(Symmetric(metric_matrix(b, ĥ)))
    @printf("  spread %.0e   λmin/λmax = %+.3e   degeneracy %.3e\n",
        spread, minimum(λ) / maximum(λ), degeneracy_residual(b, ĥ))
    record("centred 𝔻_s stays semidefinite at spread $(spread)",
        minimum(λ) > -1e-9 * maximum(λ))
end
println()

## ---------------------------------------------------------------------------------------

println(all(pass) ? "ALL CHECKS PASS" :
        "SOME CHECK FAILED ($(count(!, pass)) of $(length(pass)))")
exit(all(pass) ? 0 : 1)
