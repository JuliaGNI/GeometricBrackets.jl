# `PulledBack` against closed forms, on an annulus and on the Grad-Shafranov disk map.
#
# The pullback is the piece a mapped-domain assembly cannot be checked without, because every
# later quantity inherits its errors silently: a missing |det J| changes an area, a transposed
# J⁻¹ changes an operator, and neither raises. So it is checked here against answers known
# independently of the code.
#
# The annulus is the right first case: `F(r,θ) = (r cos θ, r sin θ)` is the same coordinate map
# as the polar disk, with the pole cut out, so a plain `TensorSplineSpace` carries it and every
# quantity has a textbook closed form.
#
#   |det J| = r                        so  ∫ 1 dμ = π(R₁² − R₀²)
#   J⁻¹J⁻ᵀ  = diag(1, r⁻²)             so  𝔻 = diag(r, r⁻¹)
#
# and the last of those is the polar Laplacian's weak form, ∫ (∂_r u ∂_r v) r + (∂_θ u ∂_θ v)/r.
#
# Then the map of `eq:mapping`, which is not a coordinate chart and has no closed-form metric,
# checked on the one number that is known independently: the area of its image, 114.777, from
# `Experiments/MetriplecticRelaxation/src/takeda.jl`.
#
# Run: julia --project=. --startup-file=no scripts/verify_pullback.jl

using LinearAlgebra
using Printf
using PoissonBrackets
# By name rather than a bare `using`: `stiffness_matrix`, `weighted_matrix` and
# `derivative_matrix` are each a different generic in the two packages, and a bare `using`
# makes all three ambiguous.
using SimpleSplines: UniformMesh, Periodic, Dirichlet, Free, (..)

const R₀ = 0.4
const R₁ = 1.3

## ---------------------------------------------------------------------------------------
## The annulus
## ---------------------------------------------------------------------------------------

Fpolar(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
DFpolar(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]

rmesh = UniformMesh(24, R₀ .. R₁)
θmesh = UniformMesh(48, 0 .. 2π)
s = TensorSplineSpace((rmesh, θmesh), 3, (Dirichlet(), Periodic()))

pb = PulledBack(s, Fpolar, DFpolar)
x̂ = quadrature_nodes(s)
w = quadrature_weights(s)

println("annulus, r ∈ [", R₀, ", ", R₁, "], ", nbasis(s), " basis functions, ",
    length(w), " quadrature nodes")
println()

## The supplied Jacobian is the right one

jr = jacobian_residual(Fpolar, DFpolar, x̂)
@printf("supplied Jacobian against a central difference   %.3e\n", jr)
jac_pass = jr < 1e-8
println("  passes: ", jac_pass)
println()

## The measure

area = dot(w, measure(pb))
area_exact = π * (R₁^2 - R₀^2)
@printf("area   ∫ 1 dμ = %.12f   exact π(R₁²−R₀²) = %.12f   relative %.3e\n",
    area, area_exact, abs(area - area_exact) / area_exact)

# The measure is |det J| = r, pointwise, not only in the integral.
m_err = maximum(abs(measure(pb)[q] - x̂[q][1]) for q in eachindex(w))
@printf("       max |m − r| pointwise = %.3e\n", m_err)
measure_pass = abs(area - area_exact) / area_exact < 1e-12 && m_err < 1e-14
println("  passes: ", measure_pass)
println()

## The metric

𝔻 = metric(pb)
d11 = maximum(abs(𝔻[1, 1][q] - x̂[q][1]) for q in eachindex(w))
d22 = maximum(abs(𝔻[2, 2][q] - 1 / x̂[q][1]) for q in eachindex(w))
doff = max(maximum(abs, 𝔻[1, 2]), maximum(abs, 𝔻[2, 1]))
@printf("metric max |𝔻₁₁ − r| = %.3e   max |𝔻₂₂ − 1/r| = %.3e   max |off-diagonal| = %.3e\n",
    d11, d22, doff)
metric_pass = max(d11, d22, doff) < 1e-13
println("  passes: ", metric_pass)
println()

## ---------------------------------------------------------------------------------------
## The weak Laplacian, and the controls that must fail
## ---------------------------------------------------------------------------------------

# For u vanishing on both circles, ∫∇u·∇v dx = −∫ v Δu dx. Both sides are assembled here: the
# left through the pullback, the right by evaluating Δu analytically and integrating against
# the pulled-back measure. They must agree to the quadrature's accuracy, and a control that
# drops a factor must not.
#
# u(r,θ) = (r−R₀)(R₁−r) cos(2θ)  vanishes on both circles.
# Δu = u_rr + u_r/r + u_θθ/r², with the θ-dependence factoring out.

u(x) = (x[1] - R₀) * (R₁ - x[1]) * cos(2 * x[2])

function Δu(x)
    r, θ = x[1], x[2]
    g = (r - R₀) * (R₁ - r)
    g′ = R₀ + R₁ - 2r
    g″ = -2.0
    (g″ + g′ / r - 4g / r^2) * cos(2θ)
end

û = project(s, u.(x̂))
v̂ = project(s, [u(pt) * (1 + 0.3 * sin(3 * pt[2])) for pt in x̂])   # a second, different field

K = tensor_weighted_matrix(s, metric(pb))
lhs = dot(v̂, K * û)

vq = basis_values(s, (0, 0))' * v̂        # `field`, which is not exported
rhs = -dot(w .* measure(pb), vq .* Δu.(x̂))

@printf("weak Laplacian   ∫∇u·∇v dμ = %.8e   −∫ v Δu dμ = %.8e   relative %.3e\n",
    lhs, rhs, abs(lhs - rhs) / abs(rhs))
laplace_pass = abs(lhs - rhs) / abs(rhs) < 1e-6
println("  passes: ", laplace_pass)

# CONTROL 1 — the plain parameter-square stiffness, with no pullback at all. It is the
# operator ∫ ∂_r u ∂_r v + ∂_θ u ∂_θ v dr dθ, which is not the Laplacian on the annulus.
#
plain = dot(v̂, PoissonBrackets.stiffness_matrix(s) * û)
@printf("  CONTROL plain stiffness, no pullback            = %.8e   relative %.3f\n",
    plain, abs(plain - rhs) / abs(rhs))

# CONTROL 2 — the measure kept but the metric dropped to |det J| times the identity, i.e. the
# 1/r² of the θθ component forgotten. This is the error that survives an area check, because
# the area never sees the metric at all.
𝔻bad = [k == l ? copy(measure(pb)) : zeros(length(w)) for k in 1:2, l in 1:2]
bad = dot(v̂, tensor_weighted_matrix(s, 𝔻bad) * û)
@printf("  CONTROL measure kept, metric dropped           = %.8e   relative %.3f\n",
    bad, abs(bad - rhs) / abs(rhs))

controls_fail = abs(plain - rhs) / abs(rhs) > 1e-2 && abs(bad - rhs) / abs(rhs) > 1e-2
println("  controls fail as they must: ", controls_fail)
println()

# 6 % and 10 % are not large numbers, and on their own they read as "nearly right". What
# separates them from the pullback's 3.5e-9 is not the size but the behaviour under
# refinement: the pullback's residual is quadrature error and converges, while a control is a
# *different operator* and converges to its own wrong answer. Measured rather than asserted.

function residuals(ncells)
    sr = TensorSplineSpace(
        (UniformMesh(ncells, R₀ .. R₁), UniformMesh(2ncells, 0 .. 2π)), 3,
        (Dirichlet(), Periodic()))
    p = PulledBack(sr, Fpolar, DFpolar)
    ŵ, x = quadrature_weights(sr), quadrature_nodes(sr)
    a = project(sr, u.(x))
    b = project(sr, [u(pt) * (1 + 0.3 * sin(3 * pt[2])) for pt in x])
    exact = -dot(ŵ .* measure(p), (basis_values(sr, (0, 0))' * b) .* Δu.(x))
    𝔹 = [k == l ? copy(measure(p)) : zeros(length(ŵ)) for k in 1:2, l in 1:2]
    (
        pullback = abs(dot(b, tensor_weighted_matrix(sr, metric(p)) * a) - exact) /
                   abs(exact),
        plain = abs(dot(b, PoissonBrackets.stiffness_matrix(sr) * a) - exact) / abs(exact),
        dropped = abs(dot(b, tensor_weighted_matrix(sr, 𝔹) * a) - exact) / abs(exact))
end

println("  under refinement — the pullback converges, the controls do not")
@printf("  %8s  %12s  %12s  %12s\n", "cells", "pullback", "plain", "metric dropped")
ref = [(n, residuals(n)) for n in (12, 24, 48)]
for (n, r) in ref
    @printf("  %8d  %12.3e  %12.3e  %12.3e\n", n, r.pullback, r.plain, r.dropped)
end
converges = ref[end][2].pullback < ref[1][2].pullback / 100
stuck = ref[end][2].plain > ref[1][2].plain / 2 &&
        ref[end][2].dropped > ref[1][2].dropped / 2
println("  pullback converges: ", converges, "        controls stay put: ", stuck)
println()

## ---------------------------------------------------------------------------------------
## A density that is not one
## ---------------------------------------------------------------------------------------

# The Grad-Shafranov measure dμ = dr dz / r, written in the physical coordinates it belongs to.
# Its pullback onto the annulus chart is |det J| / |x| = r / r = 1, so the parameter square's
# own measure comes back — a check that `density` is composed with F and not with the identity.

pbρ = PulledBack(s, Fpolar, DFpolar; density = x -> 1 / hypot(x[1], x[2]))
ρ_err = maximum(abs, measure(pbρ) .- 1)
@printf("density ρ = 1/|x| pulled back onto the chart: max |m − 1| = %.3e\n", ρ_err)
density_pass = ρ_err < 1e-13
println("  passes: ", density_pass)
println()

## ---------------------------------------------------------------------------------------
## The Grad-Shafranov disk map
## ---------------------------------------------------------------------------------------

# `eq:mapping`. Not a coordinate chart and with no closed-form metric, so what is checked is
# the Jacobian against a difference and the area against a number obtained elsewhere.
#
# The pole is cut out: a `TensorSplineSpace` cannot carry s = 0, which is the whole reason the
# polar space exists. Here that is a convenience, not an evasion — what is under test is the
# pullback algebra, and it is the same algebra at every s > 0.

const MAPC = (e = 1.4, ε = 0.3, a = 4.0, b = 3.0, c = 6.3, ξ = 1 / sqrt(1 - 0.3^2 / 4))

function disk_map(x)
    (; e, ε, a, b, c, ξ) = MAPC
    s, θ = x[1], x[2]
    q = sqrt(1 + ε * (ε + 2s * cos(θ)))
    (a * (b + (1 - q) / ε), c * e * ξ * s * sin(θ) / (2 - q))
end

# ∂/∂s and ∂/∂θ of the two components, by the chain rule through q.
function disk_jacobian(x)
    (; e, ε, a, b, c, ξ) = MAPC
    s, θ = x[1], x[2]
    q = sqrt(1 + ε * (ε + 2s * cos(θ)))
    qs = ε * cos(θ) / q
    qθ = -ε * s * sin(θ) / q
    k = c * e * ξ
    [-a*qs/ε -a*qθ/ε;
     k*(sin(θ) * (2 - q) + s * sin(θ) * qs)/(2 - q)^2 k*(s * cos(θ) * (2 - q) + s * sin(θ) * qθ)/(2 - q)^2]
end

# s from a small floor rather than 0: the tensor-product space has no pole treatment, and this
# script is testing the pullback, not the pole.
smesh = UniformMesh(64, 1e-3 .. 1.0)
sd = TensorSplineSpace((smesh, UniformMesh(128, 0 .. 2π)), 3, (Free(), Periodic()))

jr_disk = jacobian_residual(disk_map, disk_jacobian, quadrature_nodes(sd))
@printf("eq:mapping   supplied Jacobian against a central difference   %.3e\n", jr_disk)

pbd = PulledBack(sd, disk_map, disk_jacobian)
area_disk = dot(quadrature_weights(sd), measure(pbd))
const AREA_REFERENCE = 114.777
@printf("             area = %.6f   reference %.6f (takeda.jl)   relative %.3e\n",
    area_disk, AREA_REFERENCE, abs(area_disk - AREA_REFERENCE) / AREA_REFERENCE)

# The floor at s = 1e-3 removes an area of order |J| s ds ~ 1e-6 of the total, well under the
# six figures the reference is quoted to.
disk_pass = jr_disk < 1e-7 &&
            abs(area_disk - AREA_REFERENCE) / AREA_REFERENCE < 1e-5
println("  passes: ", disk_pass)
println()

## ---------------------------------------------------------------------------------------

allpass = jac_pass && measure_pass && metric_pass && laplace_pass && controls_fail &&
          converges && stuck &&
          density_pass && disk_pass
println(allpass ? "ALL CHECKS PASS" : "SOME CHECK FAILED")
exit(allpass ? 0 : 1)
