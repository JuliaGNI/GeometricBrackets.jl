```@meta
CurrentModule = GeometricBrackets
```

# GeometricBrackets.jl

Structure-preserving discretisations of the Poisson brackets of one-dimensional Hamiltonian
partial differential equations.

The package keeps the discrete bracket, the discrete Hamiltonian and the time integrator as
three separate, composable objects, so that which structure survives a discretisation can be
asked and answered one property at a time.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaGNI/GeometricBrackets.jl")
```

## The pieces

| object | what it is |
|:--|:--|
| [`DiscreteSpace`](@ref) | [`SplineSpace`](@ref) or [`LagrangeSpace`](@ref), with its quadrature |
| [`DiscreteBracket`](@ref) | ``\mathbb{P}(\hat{u})``, the discrete structure matrix |
| [`DiscreteHamiltonian`](@ref) | a functional evaluated on the discrete field, with analytic gradient |
| [`MetricBracket`](@ref) | ``\mathbb{G}(\hat{u})``, the symmetric half: positive semi-definite and degenerate on ``H`` |
| [`HamiltonianFlow`](@ref) | the two contracted: ``\dot{\hat{u}} = \mathbb{P}(\hat{u}) \, \partial H/\partial \hat{u}`` |
| [`MetriplecticFlow`](@ref) | with a dissipative half as well: ``- \mathbb{G}(\hat{u}) \, \partial S/\partial \hat{u}`` |
| [`Integrator`](@ref) | a one-step method with its nonlinear solver built once |

## A first run

```@example intro
using GeometricBrackets

sys = KdVSystem(SplineSpace(32, 3))
û₀  = project(sys.space, cosine(2π))

integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3)
traj  = integrate(sys, integ, û₀, 2000; stride = 20)

(H1 = drift(traj, :H1), H2 = drift(traj, :H2), C0 = absolute_drift(traj, :C0))
```

Implicit midpoint on the first flow is a **Poisson map** but does not conserve the cubic
``H_1``. Swapping the method for an energy-preserving one trades one for the other:

```@example intro
avf = Integrator(sys.flow1, AverageVectorField(), 1e-3)
t2  = integrate(sys, avf, û₀, 2000; stride = 20)

(H1 = drift(t2, :H1), poisson_defect_midpoint = poisson_defect(integ, copy(û₀)),
 poisson_defect_avf = poisson_defect(avf, copy(û₀)))
```

That no method here achieves both is the Ge-Marsden theorem, not a gap in the list: a
Poisson integrator that also conserved the Hamiltonian exactly would reproduce the exact flow
up to a reparametrisation of time. Its non-degeneracy hypotheses are not checked for these
systems, so it is why the trade-off is expected, not a proof that it is unavoidable here.

## What is preserved by what

| | Poisson map | generating `H` | mass |
|:--|:--|:--|:--|
| [`ImplicitMidpoint`](@ref) | yes, for a constant bracket | only if `H` is quadratic | yes |
| [`AverageVectorField`](@ref) | no | yes, for a constant bracket | yes |
| [`Gonzalez`](@ref) | no | yes, for any `H` | yes |
| [`ExplicitEuler`](@ref), [`RungeKutta4`](@ref) | no | no | yes |

The mass comes free to *every* method, explicit Euler included, because its gradient spans
the kernel of the first bracket. See [`MassCasimir`](@ref).

## The other half: discrete Lie-Poisson brackets

The pages above are one manuscript. The second asks the same question of Lie-Poisson brackets
in general — when can a discretisation satisfy the Jacobi identity *exactly* at finite ``N``? —
and the answer turns on one distinction:

| | means | delivered by |
|:--|:--|:--|
| **closure** | the discrete coefficients ``c_{ij}^m`` are the structure constants of *some* Lie algebra | Zeitlin's sine algebra, and the ``g_i \mathbb{K}_{ij} g_j`` family |
| **consistency** | they converge to the continuum commutator | conventional finite elements |

Closure is a *closed* condition: there is no second-order-accurate Jacobi identity, and a
discretisation that misses it by ``0.73`` at ``h`` misses it by ``0.73`` at ``h/8``. See
[Discrete Lie-Poisson brackets](@ref) for the structure-constant machinery and the algebra zoo
([`se3`](@ref), [`sine_algebra`](@ref), [`witt_truncation`](@ref)), and [Dirac reduction](@ref)
for what happens when the obvious repair is tried.

## The third question: Poisson brackets out of four-brackets

Both manuscripts above discretise a bracket that is already known. The third builds one, in the
continuum: feed a fixed entropy ``S`` into two of the four slots of an antisymmetric
four-bracket and ask what comes out.

On the periodic two-torus, what comes out is the Lie-Poisson bracket of two-dimensional ideal
flow — **exactly**, not to leading order, at ``S_u = \sqrt{2u}`` for the Gardner family and
``S_u = \sqrt{u}`` for the symmetric one. The Jacobi identity then holds for *every* structure
function ``c`` in ``\{A,B\}_c = \int c(u)[A_u,B_u]``, because a pointwise Plücker relation
collapses a twenty-four-term obstruction to nothing — a fact available only in two dimensions.

The difficulty is the weight. Demanding Lie-Poisson from a *weighted* bracket fixes
``\omega = 1/(2 s' s'')``, which for the Boltzmann entropy ``u \log u`` has a pole at
``u = e^{-1}``; it cancels in the two-bracket and genuinely diverges in the four-bracket, no
regular weight can replace it, and only a Casimir shift moves it out of range. See
[Poisson brackets from four-brackets](@ref), and
[The Casimir is an input, not an output](@ref) for the same pole reached from the
finite-dimensional side.

## Measuring it

Three unrelated things can go wrong with a structure-preserving discretisation, and
[Diagnostics](@ref) explains what the package measures for each: whether the *bracket* is
Poisson ([`jacobi_residual`](@ref), [`structure_constant_residual`](@ref)), whether the
*semi-discrete flow* conserves what it should ([`integrate`](@ref), [`drift`](@ref),
[`growth`](@ref)), and whether the *method* preserves the structure
([`poisson_defect`](@ref)).

## Plotting and the experiments

Load `CairoMakie` to get [`energyplot`](@ref) and [`stateplot`](@ref), and
[`sweepplot`](@ref) and [`convergenceplot`](@ref) for the refinement studies.

`scripts/kdv.jl` runs the full set of experiments:

```sh
julia --project=scripts scripts/kdv.jl          # all six cases
julia --project=scripts scripts/kdv.jl cos      # or just one
```

Per case it writes into `scripts/figures/`:

| file | contents |
|:--|:--|
| `kdv-<case>-{H1,H2,C0}-flow1.pdf` | the four flow-1 runs |
| `kdv-<case>-{H1,H2,C0}-other.pdf` | the two flow-2 runs, and the two Miura runs where the case has them |
| `kdv-<case>-state-{flow1,other}.pdf` | initial and final states, split the same way |
| `<case>.md` | a table of the maximum error in each invariant, and what it says |

One invariant per figure and one family of vector fields per figure: the three invariants'
errors sit orders of magnitude apart, so a shared axis flattens them, and the comparison that
matters is *within* a family rather than across.

`scripts/kdv_bea_sweep.jl` writes the one figure `kdv.jl` does not, the step-size sweep of the
[Backward error analysis](@ref); both take `--outdir=PATH`. The verification scripts
alongside them are indexed in [The verification scripts](@ref).

Only the `miura` case has Miura runs, and it is the only one that can: the image of the
discrete Miura map lies in ``C_{0,d} \le 0``, and the other five all have ``C_{0,d} \ge 0``
without being the zero field — ``\cos x`` has zero mass and the solitons are elevations of
positive mass. Its initial data are therefore posed in ``v``. See [`MiuraSystem`](@ref).
