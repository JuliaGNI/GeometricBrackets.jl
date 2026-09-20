```@meta
CurrentModule = GeometricBrackets
```

# Poisson brackets from four-brackets

A Poisson bracket can be manufactured by feeding a fixed entropy into two of the four slots of
an antisymmetric four-bracket. This page is what that construction yields on the periodic
two-torus: which four-brackets reduce to the Lie-Poisson bracket of two-dimensional ideal
flow, exactly rather than asymptotically; when the Jacobi identity survives; and where the
construction fails. It is `poisson-brackets-from-four-brackets.tex` and the four
`verify_fourbracket_*.jl` scripts.

!!! note "Two pages, two settings"
    [Four-brackets](@ref) on the [Discrete Lie-Poisson brackets](@ref) page asks the same
    question of a **finite-dimensional** bracket built from structure constants
    ``\mathbb{K}^{i\alpha j\beta}``, and answers it over ``\mathbb{Q}``. This page works in the
    **continuum**, with fields on a torus and a canonical bracket
    ``[f,k] = f_x k_y - f_y k_x``. The two lines of argument share no code — a
    [`TorusGrid`](@ref) is not a [`DiscreteSpace`](@ref) — and meet only in the answer, which
    is the same: the four-bracket construction reaches every Lie-Poisson bracket, and the
    weight it needs to do so is where the difficulty lives.

## The metriplectic bracket is positive, and the hypothesis is sharp

Section 2 concerns the dissipative half of a metriplectic system. Given two symmetric tensors
``\sigma`` and ``\mu``, their Kulkarni-Nomizu product

```math
R^{ijkl} = \sigma^{ik}\mu^{jl} + \sigma^{jl}\mu^{ik} - \sigma^{il}\mu^{jk} - \sigma^{jk}\mu^{il}
```

is antisymmetric in each pair of indices and symmetric under exchanging the pairs, so it is a
legitimate four-bracket. Contracting it against ``\alpha \otimes h \otimes \alpha \otimes h``
gives the induced two-bracket ``(A,A)_H`` of equation (2.6), which
[`metriplectic_bracket`](@ref) evaluates in ``O(n^2)`` rather than by building `R`.

**Proposition 2.1.** If ``\sigma`` and ``\mu`` are positive semi-definite then
``(A,A)_H \ge 0``.

The proof is Cauchy-Schwarz followed by AM-GM. `verify_fourbracket_metriplectic.jl`
cross-checks it over 800 000 draws, swept across dimension and across every rank pair
including rank-one and rank-deficient — the singular cases, where an inequality chain is
likeliest to be lost, and which a sweep over definite tensors would never reach. It also
confirms that the hypothesis is not decoration: relax ``\sigma`` to merely symmetric and a
negative value appears on the first draw.

## Two antisymmetry conditions, and why there are two families

For a four-bracket ``K(A,B;C,D)`` there are two natural antisymmetry requirements:

```math
\text{(i)}\quad K(A,B;C,D) + K(C,B;A,D) = 0 ,
\qquad
\text{(ii)}\quad K(A,B;C,D) + K(C,D;A,B) = 0 ,
```

exchanging the first and third slots, and exchanging the pair ``(1,2)`` with ``(3,4)``.
Lemma 3.1 is that the manuscript's two candidate brackets satisfy **exactly one each**:

| bracket | condition (i) | condition (ii) |
|:--|:--|:--|
| [`gardner_4bracket`](@ref) | ``0`` to roundoff | ``1.47`` |
| [`symmetric_4bracket`](@ref) | ``0.264`` | ``0`` to roundoff |

That is why the manuscript carries two families rather than one, and it is measured by
[`antisymmetry_residuals`](@ref). The non-vanishing entries matter as much as the vanishing
ones: a bracket that happened to be antisymmetric in every slot would satisfy the vanishing
half and tell you nothing, so the test fields have to be generic, with no symmetry of their
own under any exchange.

## The Gardner operators, and the identities that make the reduction work

The Gardner operators of equation (4.3) are the antisymmetric pairing of a field with one
derivative of another,

```math
G_x(f,k) = \tfrac12 (f k_x - k f_x) , \qquad G_y(f,k) = \tfrac12 (f k_y - k f_y) ,
```

and four rewritings of them carry Section 4. Two are algebraic — ``G_x(f,k) = f k_x -
\tfrac12 \partial_x(fk)``, which exhibits the operator as a flux plus a total derivative and
lets ``\int F G_x`` be integrated by parts; and ``G_x(f,k) = -\tfrac12 k^2 \partial_x(f/k)``,
the quotient form of Lemma 4.1, which turns the Gardner two-bracket into a canonical bracket
of *ratios*. The third is the curl identity

```math
\partial_y G_x(A_u,B_u) - \partial_x G_y(A_u,B_u) = -[A_u, B_u] ,
```

which is what produces a canonical bracket out of two Gardner operators in the first place.
The fourth is cyclicity of ``\int A [B, C]``, which holds because the domain is periodic.

## Reduction to Lie-Poisson form, and the factor of two

Fix the entropy: put ``S_u`` in the second and fourth slots. Then
[`gardner_2bracket`](@ref) and [`symmetric_2bracket`](@ref) *are* their four-brackets
evaluated at ``(A, S; B, S)``, which is how they are defined rather than written out again.
Both collapse to a canonical bracket weighted by a structure function, in the two available
normalisations:

```math
\{A,B\}_S^{\mathrm{Gardner}} = \tfrac12 \int S_u^2 [A_u, B_u] ,
\qquad
\{A,B\}_S^{\mathrm{sym}} = \int S_u^2 [A_u, B_u] .
```

So the target ``\int u [A_u,B_u]`` — the bracket of the ideal-fluid vorticity equation, which
[`lie_poisson_2bracket`](@ref) evaluates — is reached from the Gardner family at
``S_u = \sqrt{2u}`` and from the symmetric family already at ``S_u = \sqrt{u}``. Neither is an
approximation. The factor of two between the families is the whole difference between the two
structure functions, and it is worth stating because it is the sort of thing that a
verification script catches and a derivation does not.

Equivalently, in quotient form, ``\{A,B\}_S = \tfrac14 \int S_u^4 [A_u/S_u, B_u/S_u]``, which
is Proposition 4.3.

## Jacobi holds for every structure function

For ``\{A,B\}_c = \int c(u) [A_u, B_u]`` the Jacobi obstruction is a cyclic sum of
twenty-four terms. It collapses to nothing at all, and the mechanism is the pointwise Plücker
relation of Lemma 4.4,

```math
[A_u, B_u][C_u, u] + [B_u, C_u][A_u, u] + [C_u, A_u][B_u, u] = 0 ,
```

which [`plucker_residual`](@ref) measures. **Two dimensions is essential**: the relation is
the vanishing of a ``3\times3`` determinant of gradients that live in a two-dimensional
space, so it has no analogue in higher dimension, and neither does the theorem that rests on
it.

**Theorem 4.5.** ``\{A,B\}_c`` satisfies the Jacobi identity for *every* structure function
``c``.

[`jacobi_residual`](@ref) evaluates the obstruction on linear functionals
``A = \int \alpha u``, for which the second variations vanish identically and the cyclic sum
*is* the whole obstruction rather than a piece of it. `verify_fourbracket_identities.jl`
carries ``c(u) = u``, ``u^3``, ``\exp u``, ``\sin u`` and ``\tfrac12(1 + \log u)^2`` —
deliberately unrelated to each other, since a theorem quantified over all ``c`` is not tested
by one ``c``.

Because ``c`` is unconstrained, the Casimir family is correspondingly large: ``\int
\kappa(u)`` is a Casimir of ``\{\cdot,\cdot\}_c`` for every ``\kappa``, which is Section 4.3.

## The weighted brackets, and the singularity of the log-entropy weight

Section 5.1 puts a weight ``\omega(u)`` inside the integral —
[`weighted_2bracket`](@ref) and [`weighted_4bracket`](@ref). The two-bracket then reduces to
``2 \int \Phi(u) [A_u,B_u]`` with ``\Phi' = \omega s' s''``, so demanding Lie-Poisson fixes the
weight completely:

```math
\omega = \frac{1}{2 s' s''} .
```

For the Boltzmann entropy ``s = u \log u`` that is ``\omega = u / (2(1 + \log u))``, with a
simple pole at ``u = e^{-1}`` where ``s'`` vanishes. Three facts settle whether it can be
removed, and `verify_fourbracket_log_entropy.jl` checks each:

1. **The pole is confined to the four-bracket.** In the two-bracket the combination
   ``\omega S_u s''`` is identically ``1/2``, so the pole cancels exactly against the zero of
   ``S_u`` and the generated bracket is regular across ``u = e^{-1}`` — verified on a field
   ranging over ``[0.15, 0.65]``, which straddles it. The four-bracket, with four independent
   arguments, has no such zero available. Refinement separates the two: over ``N = 32`` to
   ``256`` the four-bracket integrand grows by ``9.7\times`` while the two-bracket integrand
   moves by ``0.9\%``.

2. **No regular weight can be substituted.** If ``s'(u_*) = 0`` and ``\omega`` is ``C^1``
   near ``u_*``, then ``c'(u_*) = 2\omega(u_*)s'(u_*)s''(u_*) = 0``, whereas Lie-Poisson needs
   ``c' \equiv 1``. So the pole is forced by the entropy, not by a poor choice of weight.

3. **It can be moved.** ``\int u`` is a Casimir of every bracket in the family, so
   ``s_\alpha = u\log u + \alpha u`` generates the same bracket while relocating the critical
   level to ``u_* = e^{-(1+\alpha)}``. On a state space whose fields take values in a compact
   subinterval of ``(0,\infty)``, some ``\alpha`` puts ``u_*`` outside it and the weight is
   smooth throughout.

!!! note "The same obstruction, from the other side"
    This is the singularity that the discrete Lie-Poisson brackets meet when the Casimir is
    prescribed rather than derived — see
    [The Casimir is an input, not an output](@ref), whose critical level is the same
    ``u = e^{-1}``, found from the structure-constant side by
    `verify_burgers_entropy_casimir.jl`. Two independent routes to one pole. What is new here
    is item 3: in the four-bracket setting the Casimir shift is available, and it is a genuine
    repair on a bounded state space rather than a relabelling.

## Section 6: a construction that generates nothing

The last section replaces the Gardner operators by their symmetric counterparts
[`symmetric_x`](@ref) and [`symmetric_y`](@ref), ``M(f,k) = f k_x + k f_x``. The resulting
two-bracket vanishes **identically**, for every ``S``, and the reason is immediate once
stated: ``M`` and ``N`` are total derivatives, so the bracket is ``\int [A_u S_u, B_u S_u]``,
the integral of a canonical bracket over a periodic domain.

The operators are nevertheless implemented as written, ``f k_x + k f_x``, and not as
``\partial_x(fk)``. Collapsing them would make the vanishing a property of the code rather
than of the bracket, and the check that asserts it would be testing nothing.

Pairing the same four derivative factors differently *does* give something, namely
``\int S_{ux} S_{uy} [A_u, B_u]``, which is the one positive residue of the section.

## How the claims are checked

Two differentiation schemes, chosen per claim, both on the periodic torus so that every
boundary term from an integration by parts vanishes.

| scheme | used for | standard of proof |
|:--|:--|:--|
| [`spectral_grid`](@ref) at ``N = 16`` | identities that are algebra plus integration by parts on band-limited fields | residual at roundoff, ``\le 10^{-11}`` |
| [`finite_difference_grid`](@ref) at ``N = 64, 128`` | anything involving ``u^{3/2}``, ``\log u``, or a quotient ``A_u/S_u`` | residual falls by ``\ge 40\times`` per doubling |

The split is not a convenience. Fourier differentiation is exact only for fields band-limited
below the Nyquist mode, and a quotient of two trigonometric polynomials is not one — so a
spectral residual for such a claim would be discretisation error misreported as a defect of
the identity. Those claims are certified by their rate instead, and
`verify_fourbracket_convergence.jl` measures observed orders of 7.2 to 8.1 against the
stencil's design order of 8. The ``40\times`` threshold sits well below the predicted
``2^8 = 256`` to absorb the constant in the leading error term, which is not close to one for
the quotient identities, while still excluding anything converging at 5th order or worse.

Test fields are fixed rather than random, and the Monte-Carlo script seeds explicitly, so a
residual that moves is a signal rather than noise.

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["torus.jl"]
```

### The brackets

```@autodocs
Modules = [GeometricBrackets]
Pages = ["fourbrackets.jl"]
```

### The metriplectic bracket

```@autodocs
Modules = [GeometricBrackets]
Pages = ["metriplectic.jl"]
```
