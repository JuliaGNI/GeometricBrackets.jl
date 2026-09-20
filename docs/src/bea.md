```@meta
CurrentModule = GeometricBrackets
```

# Backward error analysis

Implicit midpoint on the first KdV flow is a Poisson map and conserves the mass exactly. It
conserves neither Hamiltonian exactly — and yet its error in ``H_{2,d}``, the Hamiltonian it
does *not* generate, is far smaller and far better behaved than anything the method's order
would suggest, and does not grow with time. This page explains why.

It is §6 of the KdV notes, and the theory behind `scripts/verify_kdv_bea.jl`,
`scripts/kdv_bea_sweep.jl` and [`sweepplot`](@ref).

## The modified Hamiltonian

A symmetric method of order two applied to a Poisson system has a modified Hamiltonian

```math
\tilde{H} = H_{1,d} + \Delta t^2 H^{[2]}_d + O(\Delta t^4) , \qquad
H^{[2]}_d = -\tfrac{1}{24} \big\langle H''_{1,d}(\hat{u}) \, f_1(\hat{u}), \, f_1(\hat{u}) \big\rangle ,
```

with ``f_1 = \mathbb{P}^1 \partial H_{1,d}/\partial\hat{u}``, and its gradient

```math
\frac{\partial H^{[2]}_d}{\partial \hat{u}}
  = -\tfrac{1}{24}\Big( H'''_{1,d}(f_1, f_1) - 2 H''_{1,d}\,\mathbb{P}^1 H''_{1,d} f_1 \Big) .
```

The identification is not asserted: the script fits it by an order test, recovering 3.00 for the
truncation error of the modified system and 4.99 for the next correction. In the continuum the
same expansion reads

```math
\tilde{\mathcal{H}} = \mathcal{H}_1
  - \frac{\Delta t^2}{24} \int_\Omega \big( u_{tx}^2 + 6 u u_t^2 \big) \, dx + O(\Delta t^4) .
```

That accounts for ``H_{1,d}``. It does not yet explain ``H_{2,d}``, which is conserved by
nothing at all.

## ``H_{2,d}`` is the momentum of an exact symmetry

Write ``\mathbb{A} = \mathbb{M}^{-1}S``, with ``S_{kl} = \int\phi_k\partial_x\phi_l``. Four
facts, all exact at finite ``N`` and on any mesh:

```math
\mathbb{P}^1 \frac{\partial H_{2,d}}{\partial\hat{u}} = \mathbb{A}\hat{u} , \qquad
\mathbb{A}^T\mathbb{M} + \mathbb{M}\mathbb{A} = 0 , \qquad
\mathbb{A}\mathbb{P}^1 + \mathbb{P}^1\mathbb{A}^T = 0 , \qquad
\{H_{2,d}, H_{1,d}\}_{1,d} = -\mathcal{L}_\mathbb{A} H_{1,d} .
```

So ``e^{s\mathbb{A}}`` is a one-parameter group that preserves ``H_{2,d}`` and ``\mathbb{P}^1``
**exactly**, and the cross-conservation residual is precisely its defect as a symmetry of
``H_{1,d}``. ``H_{2,d}`` is its momentum map, and it is conserved to the extent that the group
is a symmetry.

The group is worth naming carefully, because the obvious candidate is the wrong one.
``e^{s\mathbb{A}}`` is the ``L^2``-**projected** translation, ``\partial_t u_h = \Pi
\partial_x u_h``, not translation itself: exact translation of a spline space by a mesh spacing
is only the finite group ``\mathbb{Z}_N`` and has no generator at all. It is the projected
version that has one, and that generator is ``\mathbb{A}``.

## The defect in closed form

```math
\varepsilon_h(\hat{u}) = 3 \, \big\langle (\mathrm{id}-\Pi)\,\partial_x u_h , \;
                                          (\mathrm{id}-\Pi)\, u_h^2 \big\rangle
```

— a pairing of **two** projection errors, and therefore far below the product of their norms
rather than of the order of either. Two properties do the remaining work:

  - **It carries no zero mode**: ``\varepsilon_h(\hat{u} + c\,e) = \varepsilon_h(\hat{u})``.
  - **It vanishes for resolved fields.** At a resolved field ``\varepsilon_h`` and
    ``\{H_{2,d}, H^{[2]}_d\}_{1,d}`` sit at round-off *together* — 1.6e-16 and -5.2e-17 — and
    both rise monotonically with the mode content, by ten orders of magnitude over the same
    sweep (``\varepsilon_h`` from 3e-12 to 5e-04). This is why a cross-conservation test on
    ``\sin x`` alone reports round-off and proves nothing.

On a uniform mesh the *quadratic* part of the residual vanishes identically, because the
assembled matrices are circulant and commute; the cubic part vanishes only for resolved fields.
On a graded mesh the quadratic part survives at ``O(h^2)``. So "exact on a uniform mesh" is a
statement about circulance, not an identity in the degrees of freedom, and
`verify_kdv_discrete.jl` §6b is the check that keeps the two apart.

## The exact increment lemma

Here the argument becomes sharp. ``H_{2,d} = \tfrac12\hat{u}^T\mathbb{M}\hat{u}`` is quadratic,
so for **any** method of the form ``\hat{u}^{n+1} = \hat{u}^n + \Delta t\,\mathbb{P}^1\bar{g}``,

```math
H_{2,d}(\hat{u}^{n+1}) - H_{2,d}(\hat{u}^n)
  = \langle \Delta\hat{u}, \mathbb{M}\bar{u} \rangle
  = -\Delta t \, \langle \bar{g}, \mathbb{A}\bar{u} \rangle ,
\qquad \bar{u} = \tfrac12(\hat{u}^n + \hat{u}^{n+1}) .
```

For the **midpoint rule**, ``\bar{g} = \partial H_{1,d}/\partial\hat{u}(\bar{u})``, and that
right-hand side is exactly ``\Delta t \, \varepsilon_h(\bar{u})``:

```math
\boxed{\;H_{2,d}(\hat{u}^{n+1}) - H_{2,d}(\hat{u}^n) = \Delta t \, \varepsilon_h(\bar{u})\;}
```

at every step size and on every mesh. The whole ``H_{2,d}`` error of the midpoint rule on the
first flow is a midpoint quadrature of a **spatial** residual. It carries no ``\Delta t``
dependence of its own — the time discretisation contributes nothing.

The energy-preserving methods do not have this property, because their increments are not the
midpoint gradient:

```math
\bar{g}_{\mathrm{AVF}} = \frac{\partial H_{1,d}}{\partial\hat{u}}(\bar{u})
  + \tfrac{1}{24} H'''_{1,d}(\Delta\hat{u},\Delta\hat{u}) , \qquad
\bar{g}_{\mathrm{G}} = \frac{\partial H_{1,d}}{\partial\hat{u}}(\bar{u})
  + \tfrac{1}{24} \frac{H'''_{1,d}(\Delta\hat{u},\Delta\hat{u},\Delta\hat{u})}
                       {\langle\Delta\hat{u},\Delta\hat{u}\rangle} \Delta\hat{u} .
```

Each correction is ``O(\Delta t^2)``, hence ``O(\Delta t^3)`` of ``H_{2,d}`` lost per step and
``O(\Delta t^2)`` over a fixed interval. That is the excess the sweep figure isolates, and the
fitted rate is 2.00 for the average vector field method and for both discrete gradients.

## Reading the sweep figure

[`sweepplot`](@ref) draws two stacked panels, and **both are needed**:

  - the **upper** panel is the departure of the respective other Hamiltonian against
    ``\Delta t``. Every curve sits within a factor of eight of the others and nothing is visible;
  - the **lower** panel is the *excess over each run's own ``\Delta t \to 0`` plateau*, taken as
    its value at the smallest step size. What is left is the ``O(\Delta t^2)`` of the
    energy-preserving methods and **nothing at all** for the midpoint rule on the first flow.

That flat line is the theorem above, drawn. A dotted ``O(\Delta t^2)`` guide is anchored to the
first series with a nonzero excess.

```sh
julia --project=scripts scripts/kdv_bea_sweep.jl [--outdir=PATH]
```

writes `kdv-bea-dtsweep.pdf`. The numbers it draws are the ones `verify_kdv_bea.jl` §6 checks —
the midpoint plateau does not move with ``\Delta t`` at all: five step sizes spanning a factor
of sixteen give the same number to four digits, a spread of 8.2e-06 about ``1.2894\times10^{-8}``.

## The plateau is spatial

Three independent confirmations, since this is the load-bearing claim:

  - it **converges under refinement**, as a spatial error must — at rates 8.7, 8.4, 8.3, 8.2 at
    ``p = 3``, better than the sixth order of the cross-conservation residual itself;
  - it is insensitive to ``\Delta t`` far past the window in which the backward-error expansion
    is valid at all;
  - its exponent is ``2p+2``, with finest-grid rates 6.16, 8.27 and 10.38 at ``p = 2, 3, 4`` —
    and the same exponent appears *twice over*, once in the dispersion error of ``\mathbb{A}``
    and once in the ``H_2`` plateau, which is what one expects if the second is a quadrature of
    the first.

## Bounded, not secular

Over a hundred time units the error oscillates and does not grow. The reason is that
``\varepsilon_h`` carries **no zero mode**, so the per-step increments do not have a constant
part to accumulate; and the only triples that would resonate are ones ``\varepsilon_h`` does not
have. This is the distinction [`growth`](@ref) exists to measure — see
[Reading a drift](@ref) — and it is why the ratio of the last tenth's envelope to the first's
is reported alongside the maximum in every table `scripts/kdv.jl` writes.

## Only one method has any of this

The argument uses, in order: that ``\mathbb{P}^1`` is constant (so ``\mathbb{A}`` exists and
``\mathbb{A}\mathbb{P}^1 + \mathbb{P}^1\mathbb{A}^T = 0``); that the method is a Poisson map (so
the modified Hamiltonian exists at all); and that ``\bar{g}`` is the midpoint gradient (so the
increment lemma closes). Only implicit midpoint on the first flow satisfies all three. On the
second flow the bracket is not constant, and the other methods are not Poisson maps —
[`poisson_defect`](@ref) is how that is checked rather than assumed. See
[Diagnostics](@ref) and [Integrators](@ref).
