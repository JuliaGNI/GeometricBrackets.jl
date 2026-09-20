```@meta
CurrentModule = GeometricBrackets
```

# Korteweg-de Vries

```math
u_t + 6 u u_x + u_{xxx} = 0
```

is bi-Hamiltonian, with

```math
\{A,B\}_1 = \int_\Omega \frac{\delta A}{\delta u} \partial_x \frac{\delta B}{\delta u} \, dx ,
\qquad H_1 = \tfrac{1}{2} \int_\Omega ( u_x^2 - 2u^3 ) \, dx
```

```math
\{A,B\}_2 = \int_\Omega \frac{\delta A}{\delta u}
    \left( -4u \partial_x - 2 u_x - \partial_x^3 \right)
    \frac{\delta B}{\delta u} \, dx ,
\qquad H_2 = \tfrac{1}{2} \int_\Omega u^2 \, dx
```

and the mass ``C_0 = \int_\Omega u \, dx`` as the Casimir of the first structure.

## The two flows are not the same scheme

[`KdVSystem`](@ref) assembles both brackets, both Hamiltonians and the two flows they
generate:

  - `flow1` = ``\mathbb{P}^1 \, \partial H_1/\partial\hat{u}`` is a *mixed* two-field
    Galerkin method, with an auxiliary field ``w_h = \Pi(3u_h^2 - u_{h,xx})`` and two mass
    solves per evaluation;
  - `flow2` = ``\mathbb{P}^2 \, \partial H_2/\partial\hat{u}`` *collapses* — the mass matrix
    in the gradient cancels one of the bracket's two inverse mass matrices — to the plain
    Galerkin discretisation, with one solve and no auxiliary variable.

They differ by exactly the projection error the first inserts between its two derivatives,
``\int \phi_i' (\mathrm{id} - \Pi)(3u_h^2 - u_{h,xx})``, which converges at ``O(h^{2p})`` —
the same order at which either scheme reproduces the equation in the first place. The choice
between them is **structural**, not a matter of accuracy: only the first is a genuine
finite-dimensional Poisson system.

```@example kdv
using GeometricBrackets
sys = KdVSystem(SplineSpace(64, 3))
û = project(sys.space, sin)
maximum(abs, vectorfield(sys.flow1, û) - vectorfield(sys.flow2, û))
```

## Sign convention

This is the textbook convention, so the solitons are **elevations**:

```math
u(t,x) = 2\kappa^2 \operatorname{sech}^2 \left( \kappa (x - 4\kappa^2 t - x_0) \right) .
```

The older ``u_t = 6uu_x - u_{xxx}`` is reached by ``u \to -u``, and each Hamiltonian carries
the matching sign, ``H_1^{\mathrm{new}}(-w) = H_1^{\mathrm{old}}(w)``. So every energy figure
in these pages is the same in either convention — the elevation soliton here has the ``H_1``
the depression soliton had there, ``-6.39998180`` at ``N = 128`` in both — while
``C_{0,d} = \int u`` changes sign. (``H_1`` itself is *not* an even functional; it is the
pairing of convention with data that is preserved.) What does move is the
second structure, and not by an overall sign: ``4u\partial_x + 2u_x - \partial_x^3`` becomes
``-4u\partial_x - 2u_x - \partial_x^3``, the third derivative keeping its sign while the
transport part flips. That asymmetry is checked symbolically in
`verify_kdv_continuous.jl` rather than asserted.

Getting the sign wrong gives a profile that steepens and blows up instead of translating.

## The Miura map

The second bracket does not need repairing. With ``u = -(v^2 + v_x)`` the Miura
factorisation

```math
(2v + \partial_x) \, \partial_x \, (2v - \partial_x) = -4u\partial_x - 2u_x - \partial_x^3
```

exhibits the second structure as the pushforward of the **first**, which is constant.
Equivalently, KdV under the second structure is mKdV under the constant one. Since
``\mathbb{P}^1`` is exactly Poisson and a Poisson structure pushed forward by a local
diffeomorphism is again Poisson, [`MiuraBracket`](@ref) inherits the Jacobi identity — at
round-off, at every resolution, against ``\approx 0.42`` flat for the Galerkin bracket:

```@example kdv
using Printf
for N in (12, 16, 24, 32)
    s = SplineSpace(N, 3)
    v̂ = project(s, miura_initial_v(2π)); û = miura_map(s, v̂)
    @printf("N = %2d   Miura %.1e   Galerkin %.3f\n", N,
            jacobi_residual(kdv_miura_bracket(s, v̂), û),
            jacobi_residual(kdv_bracket_2(s), û))
end
```

The price is locality: the Miura assembly carries two interior inverse mass matrices and is
dense, where the Galerkin one is banded. That nonlocality is not an artefact to be optimised
away — it is what the Jacobi identity costs.

### Where the chart lives

``\mathbb{P}^2_{\mathrm{M}}`` is a function of ``\hat{v}``, so writing it as a structure on
``u``-space presupposes ``\mathcal{M}_h^{-1}`` — which is undefined on most of
``\mathbb{R}^N``. Pairing ``\hat{u} = \mathcal{M}_h(\hat{v})`` with the partition of unity
gives the exact identity

```math
C_{0,d} = \int_\Omega u_h \, dx = -\int_\Omega \big( v_h^2 + \partial_x v_h \big) \, dx
        = -\int_\Omega v_h^2 \, dx ,
```

so the image lies in the half-space of **non-positive mass**, ``C_{0,d} \le 0``, with
equality only for the trivial field ``v_h \equiv 0``. The sharp criterion is spectral and
is the discrete form of the classical one: a preimage exists exactly when the Hill operator
``-\partial_x^2 - u_h`` is positive definite ([`hill_lambda0`](@ref)). Where there is one
there are two, the two Floquet solutions, with opposite ``\int_\Omega v``.

```@example kdv
shill = SplineSpace(20, 3)
uhill(c) = project(shill, x -> c + sin(x) + 0.4cos(2x))
[(c, round(hill_lambda0(shill, uhill(c)); digits = 6),
  miura_invert(shill, uhill(c)) !== nothing) for c in (0.45, 0.47, 0.48, 0.6)]
```

The consequence is concrete: at ``\lambda = 0``, ``u_0 = \cos x`` has zero mass and the
solitons are elevations of positive mass, so **none of the other examples is in the image**.
That is what [`miura_initial_v`](@ref) is for — data posed in ``\hat{v}`` from the start.

The sign convention is not what decides this. In the older convention the image was the
non-negative half-space and the solitons were excluded for being depressions; flipping to
elevations flips the half-space with them. A soliton is a reflectionless potential *with* a
bound state, and the ``\lambda = 0`` chart covers exactly the fields whose Hill operator has
none.

### The spectral parameter, which does lift it

Restore the parameter the Riccati substitution has all along, ``u = -(v^2 + v_x) - \lambda``,
and it becomes ``(-\partial_x^2 - u)\psi = \lambda\psi``: ``\lambda`` is an eigenvalue
parameter of the very Hill operator that decides invertibility, and a preimage exists exactly
when ``\lambda < \lambda_0``. For any ``\hat{u}`` that can be arranged.
[`miura_lambda`](@ref) returns such a value.

```@example kdv
sλ = SplineSpace(64, 3; L = 40.0)
û  = project(sλ, soliton(1.0, 10.0, 40.0))
(at_zero = miura_invert(sλ, û) === nothing,
 λ₀ = round(hill_lambda0(sλ, û); digits = 4),
 λ = round(miura_lambda(sλ, û); digits = 4),
 reached = miura_invert(sλ, û; λ = miura_lambda(sλ, û)) !== nothing)
```

``\mathbb{L}`` does not see a constant, so [`MiuraBracket`](@ref) is unchanged and its
Jacobiator stays at round-off for every ``\lambda``. What moves is *which* bracket the
pushforward is:

```math
\mathbb{L} \mathbb{P}^1 \mathbb{L}^T = \mathbb{P}^2 - 4\lambda \mathbb{P}^1 ,
```

a member of the bi-Hamiltonian **pencil**. On the ``u`` side the extra term is a
constant-speed translation, so the trajectory is the KdV solution in a uniformly moving frame
and all three invariants survive — ``H_2`` and ``C_0`` because the drift is generated by them,
``H_1`` because ``\{H_1,H_2\}_1 = 0``. Every case in `scripts/kdv.jl` therefore carries Miura
runs now, not only the `miura` one.

### What a time integrator does to the chart

The two systems are ``\mathcal{M}_h``-related exactly, so their *flows* are conjugate. For a
numerical method the corresponding statement is equivariance, and Runge-Kutta methods —
indeed all B-series methods — are equivariant under **affine** changes of variables and no
others. ``\mathcal{M}_h`` is quadratic, so the two computations are different methods of the
same order.

What survives a change of chart is what can be stated without one, and the Poisson-map
property is such a statement: a midpoint step in ``\hat{v}`` preserves ``\mathbb{P}^1``, and
its pushforward preserves ``\mathbb{P}^2_{\mathrm{M}}``. **This is the first genuine Poisson
integrator for a discrete second KdV structure** — the Galerkin bracket admits none.

What does not survive is the algebraic *form* of an invariant.
``H_{2,d} = \tfrac12 \hat{u}^T\mathbb{M}\hat{u}`` is quadratic in ``\hat{u}`` and **quartic**
in ``\hat{v}``, so the theorem that a symplectic Runge-Kutta method conserves quadratic
invariants applies in one chart and not the other:

| implicit midpoint on the Miura flow, ``t \le 0.2`` | ``\max\lvert\Delta H_{2,d}\rvert/\lvert H_{2,d}\rvert`` |
|:--|:--|
| in the ``\hat{v}`` chart | ``7.2\times10^{-5}`` |
| in the ``\hat{u}`` chart | ``\sim 10^{-15}`` |

Ten orders of magnitude is not a difference of implementation. What *is* chart-free is the
**value** of a function, so a discrete gradient method does conserve ``\tilde{H}``, and hence
``H_{2,d}``, exactly in the ``\hat{v}`` chart:

```@example kdv
sys = MiuraSystem(SplineSpace(20, 3))
v₀  = project(sys.space, miura_initial_v(2π))
mid = integrate(sys, Integrator(sys.flow, ImplicitMidpoint(), 1.6e-3), v₀, 125; stride = 5)
dg  = integrate(sys, Integrator(sys.flow, Gonzalez(), 1.6e-3), v₀, 125; stride = 5)
(midpoint = drift(mid, :H2), discrete_gradient = drift(dg, :H2))
```

## Initial conditions

| function | example | box | ``N`` | ``T`` | ``\Delta t`` |
|:--|:--|:--|:--|:--|:--|
| [`cosine`](@ref) | ``u_0 = \cos x`` | ``[0, 2\pi)`` | 64 | 100 | ``10^{-3}`` |
| [`soliton`](@ref) | single soliton, ``\kappa = 1``, ``x_0 = 10`` | ``[0, 40)`` | 128 | 100 | ``5\times10^{-3}`` |
| [`two_soliton`](@ref) | Hirota two-soliton, ``\kappa = 1.2, 0.8`` | ``[0, 40)`` | 128 | 100 | ``5\times10^{-3}`` |
| [`three_solitons`](@ref) | Shi-Fu-Liu Problem 4.2 | ``[-100, 100)`` | 256 | 250 | ``0.1`` |
| [`five_solitons`](@ref) | Shi-Fu-Liu Problem 4.3 | ``[-150, 150)`` | 384 | 500 | ``1/16`` |
| [`miura_initial_v`](@ref) | ``v_0 = 1 + \sin x``, posed in ``v`` | ``[0, 2\pi)`` | 64 | 100 | ``10^{-3}`` |

```@example kdv
usol = soliton(1.0, 10.0, 40.0)
usol(10.0), 2 * 1.0^2           # an elevation of height 2κ²
```

The two multi-soliton benchmarks come from Shi, Fu and Liu, *Appl. Math. Comput.* **508**
(2026) 129620. That paper writes KdV as ``u_t = \alpha u u_x + \nu u_{xxx}`` with
``\alpha = \nu = -1``; our convention is reached by ``u \to u/6``, which turns their
elevations ``12\kappa^2\operatorname{sech}^2`` into exactly the elevations above, at the
same speeds. Their momentum and energy are ``36 H_{2,d}`` and ``36 H_{1,d}``, their mass
``6 C_{0,d}``.

!!! warning "two_soliton is exact at t = 0 only"
    On the torus it carries the phase shift of *one* overtaking, whereas the two solitons
    meet again and again. Use it as initial data, not as a reference at large `t`.

Run them all with

```sh
julia --project=scripts scripts/kdv.jl
```

which writes six energy figures, a state figure and a markdown summary per case into
`scripts/figures/`. The summary tabulates the maximum error in each invariant over the run,
together with the *growth* of the error envelope — the ratio of its size over the last tenth
of the run to the first — which is what separates a bounded oscillation from a secular drift
and which no single end-of-run number can show.

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["equations/kdv.jl"]
```
