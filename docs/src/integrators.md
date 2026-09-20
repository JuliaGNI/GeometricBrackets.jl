```@meta
CurrentModule = GeometricBrackets
```

# Integrators

Three properties are in play, and no method has all three.

| method | Poisson map | generating `H` | mass |
|:--|:--|:--|:--|
| [`ImplicitMidpoint`](@ref) | yes, for a constant bracket | only if `H` is quadratic | yes |
| [`AverageVectorField`](@ref) | no | yes, for a constant bracket | yes |
| [`Gonzalez`](@ref), [`GonzalezMass`](@ref) | no | yes, for any `H` | yes |
| [`ExplicitEuler`](@ref), [`RungeKutta4`](@ref) | no | no | yes |
| [`ProjectionMethod`](@ref) | no | yes, by construction | only if asked for |

## The mass is free

Its gradient spans the kernel of the first bracket, ``\mathbb{P}^1 g = 0``, so *any*
increment lying in that bracket's range leaves it untouched — explicit Euler included. This
is the only conservation law here that costs nothing.

## Energy or structure, but not both

That no method here is both a Poisson map and exactly energy-preserving is the
**Ge-Marsden theorem**: such a method would reproduce the exact flow up to a reparametrisation
of time. The theorem's non-degeneracy hypotheses are not checked for these systems, so it is
the reason to expect the trade-off rather than a proof of it here. The trade-off itself is
measured directly, through [`poisson_defect`](@ref).

## Why the tangent map is analytic

[`tangent_map`](@ref) differentiates the step *implicitly* rather than by finite
differences. A central difference has a floor around ``4\times10^{-10}`` on these problems,
while the defect of a genuine Poisson integrator sits at round-off — so a differenced
tangent map would measure its own truncation error and report it as a violation of the
Poisson property, three orders too large.

For implicit midpoint the implicit differentiation gives the Cayley transform

```math
D\Phi_{\Delta t} = \left( \mathbb{I} - \tfrac{1}{2}\Delta t \, \mathbb{P} H'' \right)^{-1}
                   \left( \mathbb{I} + \tfrac{1}{2}\Delta t \, \mathbb{P} H'' \right) ,
```

and a Cayley transform of ``\mathbb{P}`` times a symmetric matrix preserves ``\mathbb{P}``
exactly.

## The nonlinear solve

[`Integrator`](@ref) builds one `SimpleSolvers.NewtonSolver` and reuses it across every
step. Its `refactorize` option is the quasi-Newton scheme these problems want: the Jacobian
is factorised once and reused for several iterations.

The Jacobian supplied is analytic, including for the Gonzalez discrete gradient, whose
rank-one correction is differentiated in closed form.

Two choices about the solve are worth knowing about, both measured rather than assumed:

  - **No line search.** The residual is a small perturbation of the identity at these step
    sizes and Newton converges in two or three iterations from the previous state, so the
    default `Backtracking` spends several extra residual evaluations per iteration probing a
    step length that is always accepted at one. The default here is `Static`. Pass
    `linesearch = Backtracking(Float64)` if a run is pushed to a step size that needs it.
  - **A LAPACK-backed factorisation.** The factorisation is left to SimpleSolvers, which
    since 0.13 selects `LapackLU` for a dense Jacobian — it used to default to a hand-written
    scalar LU, portable and the right choice for the small dense systems it is usually
    pointed at, but at ``N = 384`` that accounted for **74 % of the cost of an implicit
    step**, about 17 ms against LAPACK's 0.6 ms. LAPACK is restricted to the element types it
    provides, so `linear_solver_method = SimpleSolvers.LU()` remains the only option for
    e.g. `BigFloat`.

Together with the assembly changes of the [Discretisation](@ref) page this took an implicit
step at ``N = 384`` from 45 ms to 4.2 ms.

## The mixed two-field formulation

The Newton matrix of the dense formulation is dense, and it is dense for one reason: the
Poisson matrix is the sandwich ``\mathbb{M}^{-1} \mathbb{K}(\hat{u}) \mathbb{M}^{-1}``, and
``\mathbb{M}^{-1}`` of a banded mass matrix genuinely is full. Every factor around it is
banded. `formulation = :mixed` never forms that sandwich, by carrying the intermediate as an
unknown of its own — solving ``2N`` equations in ``(y, v)`` rather than ``N`` in ``y``:

```math
\begin{aligned}
\mathbb{M} (y - \hat{u}^n) - \Delta t \, \mathbb{K}(\bar{u}) \, v &= 0, \\
\mathbb{M} v - \partial H/\partial u(\bar{u}) &= 0,
\end{aligned}
\qquad \bar{u} = \tfrac{1}{2}(\hat{u}^n + y).
```

Eliminating ``v`` from the first equation recovers the dense residual exactly, up to a factor
of the mass matrix. It is not an approximation of the dense step, or a different method that
happens to be close to it — it is the **same step**, and the test suite holds the two to that,
to 12–14 digits over single steps and over 100-step runs, conserved quantities included. See
[`mixed_residual!`](@ref).

What changes is the linear algebra. All four blocks of the mixed Jacobian are built from
``\mathbb{M}``, ``\mathbb{K}``, ``\partial(\mathbb{K}v)/\partial u`` and ``\partial^2 H`` —
each banded with circular bandwidth ``p``, none of them requiring ``\mathbb{M}^{-1}``. The
weak-form block is exposed as [`kernel_operator`](@ref), which is [`poisson_matrix`](@ref)'s
middle factor and is sparse where `poisson_matrix` is dense. Better still, the sparsity pattern
is fixed for the whole run, so one ordering and one symbolic factorisation serve every step.

It is not free: ``2N`` unknowns against ``N``, so the dense form wins while ``N`` is small.
One step of implicit midpoint on the second KdV flow at ``p = 3``, in milliseconds, both
formulations converging in two Newton iterations:

| ``N`` | dense | mixed | speedup |
|---:|---:|---:|---:|
| 64 | 0.192 | 0.259 | 0.74 |
| 128 | 0.518 | 0.485 | 1.07 |
| 384 | 3.34 | 1.56 | **2.1** |
| 1024 | 22.9 | 4.22 | **5.4** |
| 1536 | 58.2 | 6.29 | **9.3** |

The crossover is near ``N = 125``, so `:dense` remains the default and the small cases are
untouched. The gain widens because the dense factorisation is ``O(N^3)`` and the assembly's
two dense ``N \times N`` products are too, where the mixed form is ``O(N)`` in both.

Two restrictions, and both are an `ArgumentError` from [`Integrator`](@ref) rather than a
silently different method:

  - **[`ImplicitMidpoint`](@ref) only.** [`AverageVectorField`](@ref) would need one auxiliary
    per quadrature node, and the discrete-gradient methods carry a rank-one term in
    ``\partial \bar{g}/\partial y`` that would have to be bordered.
  - **[`AffineBracket`](@ref) only** — the KdV and Camassa-Holm second brackets. The other
    brackets store the sandwiched ``\mathbb{M}^{-1}\mathbb{K}\mathbb{M}^{-1}``, formed once in
    the constructor, so ``\mathbb{K}`` cannot be recovered from them without inverting the
    mass matrix again.

!!! warning "The linear solver is a correctness constraint here"
    The mixed system is block-structured, and UMFPACK mishandles it. From ``N = 768`` upward
    `SimpleSolvers.UmfpackLU` returns a solution wrong by a factor of 150 *while reporting
    success*, which surfaces as Newton diverging from a starting point whose residual was
    already ``10^{-5}``. `SparspakLU` and dense LAPACK both solve the same matrices to the
    accuracy their condition number allows, so `SparspakLU` — pure Julia, and light — is the
    default for this formulation, in place of the `UmfpackLU` that SimpleSolvers would
    otherwise select for a sparse `Float64` Jacobian. Do not override it without checking the
    residual.

## Measuring what survives

[`integrate`](@ref) evaluates the invariants at **every** step but reports **windowed
maxima**, [`drift`](@ref) and [`absolute_drift`](@ref) are not interchangeable, and
[`growth`](@ref) is what separates a bounded oscillation from a secular drift. What each of
those measures, and how to tell a drift from accumulated round-off, is on the
[Diagnostics](@ref) page.

The *size* of the one error this table does not account for — the respective other Hamiltonian
along a Poisson integrator — is not arbitrary either. It has a closed form, and it is the
subject of [Backward error analysis](@ref).
