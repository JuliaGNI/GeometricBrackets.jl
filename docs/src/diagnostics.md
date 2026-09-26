```@meta
CurrentModule = GeometricBrackets
```

# Diagnostics

A discretisation of a Hamiltonian PDE can fail in three unrelated ways, and this package
measures them with three unrelated sets of numbers. Keeping them apart is most of the work,
because a claim about one is routinely offered as evidence for another:

| question | about | measured by |
|:--|:--|:--|
| Is the discrete bracket a **Poisson** bracket? | the bracket alone | [`isantisymmetric`](@ref), [`jacobi_residual`](@ref), [`structure_constant_residual`](@ref) |
| Does the **semi-discrete flow** conserve what it should? | bracket ∘ Hamiltonian | [`invariants`](@ref), [`integrate`](@ref), [`drift`](@ref), [`growth`](@ref) |
| Does the **method** preserve the structure? | the time discretisation | [`poisson_defect`](@ref), [`tangent_map`](@ref) |

None of the three implies another. A bracket can be exactly antisymmetric, and hence conserve
its generating Hamiltonian to the last bit, while failing the Jacobi identity at order one —
that is precisely what the second KdV bracket does. A method can be a Poisson map and lose the
energy, or conserve the energy and destroy the bracket. And a quantity can be conserved
because the arithmetic happens to be small rather than because anything preserves it.

## Structural: is the bracket a Poisson bracket?

### Antisymmetry is free, so it is not a result

Every bracket here is assembled in explicitly skew-symmetrised form, so
[`isantisymmetric`](@ref) holds to the last bit at any number of quadrature points and on any
mesh. A nonzero value is a bug, not a finding.

This is worth having made free. The unsymmetrised assembly of the second KdV bracket is the
*same matrix* in exact arithmetic — the term that differs is symmetric in ``k \leftrightarrow
l`` and drops out of the skew part — but it is antisymmetric only if the quadrature integrates
a total derivative of degree ``3p-1`` exactly. Antisymmetry alone gives

```math
\dot{H} = \left( \frac{\partial H}{\partial \hat{u}} \right)^{\!T} \mathbb{P}
          \left( \frac{\partial H}{\partial \hat{u}} \right) = 0 ,
```

so exact conservation of the generating Hamiltonian comes with it, at every quadrature. See
[Why the skew-symmetrised form matters](@ref).

### The Jacobiator, and why it is normalised

[`jacobi_residual`](@ref) is

```math
\frac{\max_{ijk} \left| \sum_l \left(
      \mathbb{P}_{il} \, \partial_l \mathbb{P}_{jk}
    + \mathbb{P}_{jl} \, \partial_l \mathbb{P}_{ki}
    + \mathbb{P}_{kl} \, \partial_l \mathbb{P}_{ij} \right) \right|}
     {\max_{ijk} \left| \sum_l \mathbb{P}_{il} \, \partial_l \mathbb{P}_{jk} \right|} ,
```

the Jacobiator divided by **the largest single term entering it**. The choice of denominator
is the whole point. A raw residual can be driven to zero by any change that merely shrinks the
bracket; one normalised by ``\max|\mathbb{P}|^2`` by any change that makes it blow up. Dividing
by the largest term actually formed is neither, and it is what makes the number comparable
across resolutions.

```@example diag
using GeometricBrackets, Random
Random.seed!(1)

s = SplineSpace(16, 3)
û = randn(16)
(constant = jacobi_residual(kdv_bracket_1(s), û),
 affine   = jacobi_residual(kdv_bracket_2(s), û))
```

The method is generic in the element type. Over `Rational{BigInt}` a vanishing residual is
`iszero` and not "below a tolerance someone chose" — which is why every structural claim in
`scripts/` that can be posed exactly is posed exactly.

### Linear brackets: the structure-constant condition

A bracket **linear** in the field, ``\mathbb{J}_{ij}(u) = \sum_m c_{ij}^m u_m``, is Poisson if
and only if the ``c_{ij}^m`` are the structure constants of a Lie algebra,

```math
\sum_m \left( c_{ij}^m c_{mk}^n + c_{jk}^m c_{mi}^n + c_{ki}^m c_{mj}^n \right) = 0
\qquad \forall\, i,j,k,n ,
```

which is [`structure_constant_residual`](@ref), normalised the same way. An
[`AffineBracket`](@ref) ``\mathbb{P}(\hat{u}) = \mathbb{P}^0 + \sum_m \hat{u}_m C^m`` therefore
splits its Jacobiator by degree in ``\hat{u}``: a degree-two Lie-algebra condition on the
``C^m``, and a degree-one **2-cocycle** condition pairing ``\mathbb{P}^0`` against them. For
the second KdV bracket the two sit at ``0.4212`` and ``0.4665``, and neither moves under
refinement. (Those are conditions on the coefficients, so they are single numbers; the
``0.43`` above is [`jacobi_residual`](@ref) of the assembled bracket *at a point*, which varies
with the point and not with ``N``.)

!!! warning "Choosing a control"
    Do **not** validate a routine like this against ``\mathfrak{so}(3)``. It lies in the
    six-parameter family ``c_{ij}^k = \epsilon_{ijl} n^{lk}`` with ``n`` symmetric — Bianchi
    class A — every member of which satisfies the condition, and the perturbations one reaches
    for (rescaling a generator, rescaling one structure constant) stay inside it. So it goes on
    passing after it looks broken.

    It is *not* true that antisymmetry alone forces the Jacobi identity in dimension three: a
    general antisymmetric `c` has nine parameters against this family's six, and random ones
    fail comfortably. Use [`se3`](@ref) as the positive control and
    [`random_antisymmetric_c`](@ref) in dimension five as the negative one.

```@example diag
rng = MersenneTwister(5)
(se3     = structure_constant_residual(se3()),
 so3     = structure_constant_residual(so3()),          # passes, and tells you nothing
 random5 = structure_constant_residual(random_antisymmetric_c(rng, 5)))
```

### A flat residual is the answer, not a failure to converge

The structure-constant condition is a **closed** condition on the coefficients. There is no
such thing as a second-order-accurate Jacobi identity: either the coefficients close into a Lie
algebra or they do not, and a discretisation that misses by ``0.7`` at ``h`` misses by ``0.7``
at ``h/8``. Two requirements have to be kept apart:

  - **(H) closure** — the ``c`` are the structure constants of *some* Lie algebra. This is what
    makes the bracket Poisson, exactly, at every ``N``.
  - **(C) asymptotic homomorphism** — the coefficients converge to the continuum commutator.
    This is what makes the scheme consistent.

Conventional finite elements deliver (C) and not (H), and (H) cannot be approximated:

```@example diag
using Printf
for ne in (8, 16, 32)
    sp   = LagrangeSpace(1, ne); N = nbasis(sp)
    Minv = inv(Matrix(mass_matrix(sp)))
    b    = basis_integrals(sp)
    E    = nodal_derivative_matrix(sp)                  # E[q, m] = φ_q'(x_m)
    A, B = Minv * E, transpose(E) * Minv
    C    = [b[m] * (Minv[i,m] * B[m,j] - A[i,m] * Minv[m,j]) for m in 1:N, i in 1:N, j in 1:N]
    @printf("N = %3d   residual = %.4f\n", N, structure_constant_residual(C))
end
```

See [Discrete Lie-Poisson brackets](@ref) for what does close, and
[Aliasing and the zero-mode theorem](@ref) for why the obvious repair cannot be made to.

### The contracted Jacobiator asks a different question

The normalised residual asks whether the discrete bracket is a Poisson bracket. Contracting the
Jacobiator against three *fixed smooth* functionals asks whether the discrete bracket
approximates a continuum identity that does hold. These are different questions and they get
different answers: for cubic splines the contracted quantity converges at better than fourth
order while the normalised residual does not converge at all. Both statements are true; a
claim has to say which one it is about.

### Casimirs, exactly

The Casimirs of a discrete bracket are the kernel of its structure matrix, so finding them is a
rank question — and `LinearAlgebra.rank` and `nullspace` route through the SVD and do not run
over ``\mathbb{Q}``. [`rref`](@ref), [`exact_rank`](@ref) and [`kernel`](@ref) do.

Two traps live here:

1. **An antisymmetric matrix has even rank.** With an odd number of degrees of freedom the
   corank is one and the single kernel vector is the physical Casimir; with an even number there
   is a second, the sawtooth ``(-1)^i``, and a *spurious* discrete Casimir with no continuum
   counterpart. Choose ``N`` odd.
2. **Uniform meshes confirm identities for the wrong reason** — their assemblies are circulant.
   Use a `RandomMesh` for anything claimed to hold on any mesh.

### Closure of a truncation

[`closes_on`](@ref) and [`leak`](@ref) measure whether a coarse block ``V_1`` closes on
``V_1 \oplus V_2``, ``\max|c_{ij}^m|`` over ``i,j \in V_1`` and ``m`` outside. This is the
premise the hierarchical-basis construction of [Dirac reduction](@ref) needs, and it is a
separate question from whether either block is a Lie algebra:

```@example diag
w = witt_truncation(2)      # |k| ≤ 2 kept, 2 < |k| ≤ 4 constrained
(closes = closes_on(w.C, w.keep, w.con),
 leak   = leak(w.C, w.keep, w.con),
 jacobi = structure_constant_residual(w.C))
```

The truncation closes exactly and is not a Lie algebra. [`jacobiator`](@ref) and
[`project_jacobiator`](@ref) are what carry that residual through a Dirac reduction; see
[Dirac reduction](@ref).

## Conservation: what the semi-discrete flow keeps

### What is guaranteed, and by what

| quantity | why it is conserved | how well |
|:--|:--|:--|
| the generating Hamiltonian | antisymmetry of ``\mathbb{P}``, nothing else | exactly, at any quadrature and on any mesh |
| the mass | ``\mathbb{P}^1 g = 0`` with ``g_i = \int\phi_i`` — see [`MassCasimir`](@ref) | exactly, for *every* method including explicit Euler |
| the respective other Hamiltonian | **nothing** | to the order of the discretisation |

The third row is the interesting one, and it is what a `drift` on ``H_2`` along the first flow
actually measures. Its size is not arbitrary: it has a closed form, and it is the subject of
[Backward error analysis](@ref).

### The invariants, and why they are computed twice

[`invariants`](@ref) returns a `NamedTuple` assembled from a **single** evaluation of the field
and its derivative rather than by calling each [`DiscreteHamiltonian`](@ref) in turn — it runs
once per step, and at the step counts these runs reach it is a third of the total time.

!!! note "A deliberate duplication"
    That means `invariants(::KdVSystem, …)` re-implements ``H_1`` instead of calling
    `hamiltonian(KdVHamiltonian1(), …)`, and the two have to be changed together. This is not
    an oversight, and the risk it carries is recorded rather than hidden: when the sign
    convention was flipped, leaving one copy stale made implicit midpoint report a 373 % drift
    in a quantity the flow conserves exactly. A sign that moves in one copy and not the other is
    invisible to every *structural* test and shows up only as a drift, which is why
    `test/diagnostics.jl` holds the fused values against the individual Hamiltonians.

[`invariants`](@ref) of a [`MiuraSystem`](@ref) reports **in ``u``**: the state is ``\hat{v}``,
so the map through [`miura_map`](@ref) comes first and a Miura run can be compared directly
against the ``u``-chart ones.

### Windowed maxima, not samples

[`integrate`](@ref) evaluates the invariants at **every** step and reports the largest
deviation reached anywhere inside each output window. `deviation[k, j]` is a maximum over
window `j`, not its final value.

This is not a convenience. The error envelope of a geometric integrator oscillates with a
period of tens of steps. Sampling it every `stride` steps over a run of millions aliases that
oscillation into noise, and the noise can sit an order of magnitude below the true envelope —
so a figure built from samples shows a method conserving an invariant far better than it does.
The invariants are therefore evaluated at every step and only the *reporting* is windowed.

```@example diag
s2   = KdVSystem(SplineSpace(16, 3))
u1   = project(s2.space, x -> sin(x) + 0.4cos(2x))
mkstep() = Integrator(s2.flow1, ImplicitMidpoint(), 2e-3)

fine = integrate(s2, mkstep(), u1, 400; stride = 1)     # every step
wind = integrate(s2, mkstep(), u1, 400; stride = 20)    # windowed maxima
sampled = deviation(fine, :H1)[20:20:400]             # what subsampling would report

maximum(deviation(wind, :H1) ./ sampled)
```

### Reading a drift

```@example diag
sys  = KdVSystem(SplineSpace(24, 3))
û₀   = project(sys.space, cosine(2π))
traj = integrate(sys, Integrator(sys.flow1, ImplicitMidpoint(), 1e-3), û₀, 2000; stride = 20)

(H1 = drift(traj, :H1), C0 = absolute_drift(traj, :C0), growth_H1 = growth(traj, :H1))
```

  - [`drift`](@ref) is the largest **relative** deviation, ``\max_j |I - I_0|/|I_0|``. Use it
    for the Hamiltonians.
  - [`absolute_drift`](@ref) is for an invariant whose reference value vanishes. The mass of the
    cosine initial condition is zero, and numerically about ``10^{-17}``; dividing a ``10^{-15}``
    deviation by that would report a hundredfold "relative drift" of a quantity conserved to
    round-off. `drift` therefore **throws** rather than returning a number that would be read as
    a result.
  - [`growth`](@ref) is the ratio of the error envelope over the last tenth of the run to that
    over the first. Near one is a bounded oscillation; large is a secular drift. This is the
    number that separates "``H_1`` departs by ``10^{-6}`` and stays there" from "``H_1`` is being
    lost", and the two are indistinguishable in a single end-of-run figure.

### Telling round-off from dynamics

A drift is only evidence if it is above the arithmetic floor, and the package carries its own
meter for that floor. ``C_0`` is conserved **exactly** in exact arithmetic, by
[`MassCasimir`](@ref), so its measured deviation is nothing but accumulated round-off — 4.7e-12
over the 766781 steps of the explicit cosine run. Any ``|\Delta H|`` reported at that same order
is arithmetic and not dynamics. Two consequences worth stating:

  - a large [`growth`](@ref) on an error still at round-off is only the random walk of the
    arithmetic;
  - two implementations agreeing on the dynamical columns and differing on the round-off-limited
    ones agree. That is the whole content of [The Runge-Kutta 4 question](@ref).

## Geometric: what the method preserves

### The Poisson-map defect

A map ``\Phi`` is Poisson when ``D\Phi \, \mathbb{P} \, D\Phi^T = \mathbb{P}``, and
[`poisson_defect`](@ref) is the relative failure of that identity,
``\|D\Phi\,\mathbb{P}\,D\Phi^T - \mathbb{P}\| / \|\mathbb{P}\|``:

```@example diag
Δt  = 1e-3
mid = Integrator(sys.flow1, ImplicitMidpoint(),    Δt)
avf = Integrator(sys.flow1, AverageVectorField(),  Δt)
dgr = Integrator(sys.flow1, Gonzalez(),            Δt)
eul = Integrator(sys.flow1, ExplicitEuler(),       Δt)

[nameof(typeof(i.method)) => poisson_defect(i, copy(û₀)) for i in (mid, avf, dgr, eul)]
```

Only the first is at round-off, and only the first is a Poisson map. The defect of a method
that is *not* one is ``O(\Delta t)``, so these numbers are comparable to each other at a fixed
step size and not across step sizes; what is scale-free is the gap — fourteen orders of
magnitude from the midpoint rule to explicit Euler, and ten to the nearest non-Poisson method.

### Why the tangent map must be analytic

[`tangent_map`](@ref) differentiates the step *implicitly*. A central difference has a floor
around ``4\times10^{-10}`` on these problems, three orders **above** the defect of a genuine
Poisson integrator — so a differenced tangent map would measure its own truncation error and
report it as a violation of the Poisson property. For implicit midpoint the implicit
differentiation gives the Cayley transform

```math
D\Phi_{\Delta t} = \left( \mathbb{I} - \tfrac{1}{2}\Delta t \, \mathbb{P} H'' \right)^{-1}
                   \left( \mathbb{I} + \tfrac{1}{2}\Delta t \, \mathbb{P} H'' \right) ,
```

and a Cayley transform of ``\mathbb{P}`` times a symmetric matrix preserves ``\mathbb{P}``
exactly.

### Why no method scores on both counts

That no method here is both a Poisson map and exactly energy-preserving is the **Ge-Marsden
theorem**: such a method would reproduce the exact flow up to a reparametrisation of time. Its
non-degeneracy hypotheses are not checked for these brackets, so it is the reason to *expect*
the trade-off rather than a proof of it here — and the trade-off itself is measured, not
assumed.

A second, sharper obstruction explains why implicit midpoint loses even the **quadratic**
``H_2``. Cooper's theorem gives a symplectic Runge-Kutta method a quadratic invariant
``Q(y) = y^T C y`` only when ``Q'(y) f(y) = 0`` for **all** ``y``, not merely along solutions.
Switching off the cubic term restores it, and so does making the mesh uniform; each alone is
enough to break it.

### Stability, and what an explicit method costs

`stability_limit(flow, û)` is ``2\sqrt{2}/\rho`` with ``\rho`` the spectral radius of the
linearised flow. Both fields here have essentially imaginary spectra, so the imaginary-axis
limit is the relevant one, and ``\rho`` grows like ``h^{-3}`` through the third derivative —
which is why an explicit method needs so many more steps at the same resolution. The measured
values ``\rho = 115.4, 274.2, 907.2, 2170.0`` at ``N = 12, 16, 24, 32`` are asserted by the test
suite, so a change in the assembly fails a test rather than drifting away from the manuscript.
``\rho`` is sensitive to the amplitude, which is why the initial datum has to be stated
alongside it.

### Buying every invariant, and what it costs

[`ProjectionMethod`](@ref) follows a base method with a projection onto the joint level set of
several invariants, solving a small nonlinear system for the multipliers. It holds all of them
by construction and is not a Poisson map. One trap: the correction direction of two Hamiltonians
is not mass-neutral, so **pass the mass among the invariants** if it is to be kept — projecting
onto the two Hamiltonians alone loses the Casimir that every other method here gets for free.

## Refinement studies

[`convergenceplot`](@ref) draws `label => errors` against a resolution on log-log axes and fits
an order per series. It was written as much for the claims where a rate is *not* achieved as for
those where it is: a residual that is flat under refinement — the ``\approx 0.42`` Jacobiator of
the second KdV bracket, the ``\approx 0.73`` structure-constant violation of the nodal elements —
reads as a horizontal line against the sloped ones, which is the honest way to show that no
amount of refinement will help.

The rates worth knowing, and where each comes from:

| quantity | order | why |
|:--|:--|:--|
| difference between the two KdV flows | ``2p`` | the projection error ``\int \phi_i' (\mathrm{id}-\Pi) r_h`` the first inserts |
| the ``H_2`` plateau, and the dispersion error of ``\mathbb{A}`` | ``2p+2`` | see [Backward error analysis](@ref) |
| quadrature needed for consistency | degree ``3p-1`` | the cubic term of the second bracket |
| quadrature needed for the mass | degree ``2p-1`` | ``\int \partial_x(u_h^2) = 0`` |
| the sine algebra towards ``m \times n`` | ``N^{-2}`` | asymptotic in ``N \gg 2\pi (m\times n)``; at ``N = 11`` the observed order is 0.8 |
| the second Magri rung | ``O(h^{2p})`` | the same signature as the Jacobi failure |

## Plotting

Load `CairoMakie` to get [`energyplot`](@ref), [`stateplot`](@ref), [`sweepplot`](@ref) and
[`convergenceplot`](@ref).

The encoding is deliberate and worth keeping if these are adapted: **colour carries the
integrator** ([`INTEGRATOR_COLORS`](@ref), Okabe-Ito) and **line style carries the vector field**
([`FLOW_STYLES`](@ref): solid for the first flow, dashed for the second, dash-dotted for the
Miura flow). No series is identified by colour alone, so the figure survives being printed in
grey.

Three further choices, each of which changes what a reader concludes:

  - **One invariant per figure, one family of vector fields per figure.** The three invariants'
    errors sit orders of magnitude apart, so a shared axis flattens them; and the comparison that
    matters is *within* a family rather than across.
  - **[`ERROR_FLOOR`](@ref) clamps rather than drops.** A deviation below round-off is drawn at
    ``10^{-16}``, so a method conserving an invariant exactly shows as a flat line at the floor
    instead of a gap in the curve.
  - **A vanishing reference switches the panel to absolute error**, and says so on the axis,
    for the reason [`drift`](@ref) throws.

`scripts/kdv.jl` writes the whole set of figures, and `scripts/kdv_bea_sweep.jl` the step-size
sweep; both take `--outdir=PATH`. See [The verification scripts](@ref).

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["diagnostics.jl"]
```
