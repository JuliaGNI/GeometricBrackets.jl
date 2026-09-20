```@meta
CurrentModule = GeometricBrackets
```

# Camassa-Holm

!!! note "Prototype"
    Camassa-Holm accompanies neither manuscript. It is here because the same construction
    applies, and is tested to the same standard, but no claim beyond those tests is made.

```math
m_t + u m_x + 2 u_x m = 0 , \qquad m = u - u_{xx} ,
```

is bi-Hamiltonian with

```math
B_1 = \partial_x - \partial_x^3 ,
\qquad H_2 = \tfrac{1}{2}\int_\Omega (u^3 + u u_x^2) \, dx
```
```math
B_2 = m\partial_x + \partial_x m ,
\qquad H_1 = \tfrac{1}{2}\int_\Omega (u^2 + u_x^2) \, dx .
```

## The evolved variable is the momentum

This is not presentational. The Camassa-Holm bi-Hamiltonian structure is written with
variational derivatives with respect to ``m``, and taking them with respect to ``u`` instead
gives two flows that differ by tens of per cent and **do not converge to one another** —
which is exactly how the error was caught.

The velocity is recovered by the discrete Helmholtz solve of [`HelmholtzMap`](@ref),

```math
\mathbb{A} \hat{u} = \mathbb{M} \hat{m} , \qquad \mathbb{A} = \mathbb{M} + \mathbb{K}^1 ,
```

with ``\mathbb{A}`` the ``H^1`` Gram matrix, symmetric positive definite. Use
[`velocity`](@ref) and [`momentum`](@ref) to move between the two.

```@example ch
using GeometricBrackets
sys = CamassaHolmSystem(SplineSpace(48, 3))
m̂ = momentum(sys, project(sys.space, x -> 1 + 0.5sin(x)))
maximum(abs, vectorfield(sys.flow1, m̂) - vectorfield(sys.flow2, m̂))
```

## Structure

[`camassa_holm_bracket_1`](@ref) is constant and exactly Poisson.
[`camassa_holm_bracket_2`](@ref) is the same [`AffineBracket`](@ref) as the second KdV
bracket with `scale` one instead of two, and like it does **not** satisfy the Jacobi
identity — ``m\partial_x + \partial_x m`` is again a Lie-Poisson structure of Virasoro type,
and the residual is of order one and flat under refinement.

!!! note "The mass is not a Casimir of the second bracket"
    ``\int_\Omega m \, dx`` is Casimir-strength under ``\mathbb{P}^1``, where
    ``\mathbb{P}^1 g = 0`` identically. Under ``\mathbb{P}^2`` it is conserved only to the
    order of the discretisation: its continuous conservation rests on ``\int u_x m \, dx =
    0``, which holds because both terms of ``u_x(u - u_{xx})`` are total derivatives — a
    statement about what the discretisation resolves, not an identity in ``\hat{m}``. KdV,
    where the mass survives under both brackets, is the more fortunate case rather than the
    general one.

## Initial conditions

A smooth positive profile such as ``u_0 = 1 + \tfrac{1}{2}\sin x`` on ``[0, 2\pi)`` with
``p = 3`` and ``N = 24`` is what the tests use. Remember to convert it:

```@example ch
m̂₀ = momentum(sys, project(sys.space, x -> 1 + 0.5sin(x)))
traj = integrate(sys, Integrator(sys.flow2, ImplicitMidpoint(), 1e-3), m̂₀, 300; stride = 10)
drift(traj, :H1)          # quadratic in m̂, so a symplectic method holds it exactly
```

The other standard datum is the [`peakon`](@ref) ``u = c\,e^{-|x-x_0|}``, a travelling wave
of speed `c`.

!!! warning "The peakon is a weak solution"
    It has a corner at its crest, so ``u_x`` jumps there and the momentum ``m = u - u_{xx}``
    is a delta. A spline discretisation cannot represent that, and the momentum it computes
    is a smoothed version of the delta — worth knowing before reading anything into a
    convergence study run on this initial condition.

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["equations/camassaholm.jl"]
```
