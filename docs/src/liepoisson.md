```@meta
CurrentModule = GeometricBrackets
```

# Discrete Lie-Poisson brackets

The second manuscript asks a question the [Korteweg-de Vries](@ref) pages leave open. There,
the second structure fails the Jacobi identity and the
[Miura chart](@ref "The Miura map") is what repairs it. Here the failure is examined in general: *when* can a discretisation of a
Lie-Poisson bracket be Poisson at finite ``N``, and what does it cost?

The answer turns on a single distinction, and most of this page is spent on it.

## The bracket that started it

The one-dimensional transport bracket

```math
\{A,B\} = \int_\Omega u \left(
    \frac{\delta A}{\delta u} \partial_x \frac{\delta B}{\delta u}
  - \frac{\delta B}{\delta u} \partial_x \frac{\delta A}{\delta u} \right) dx
```

is a Lie-Poisson bracket for ``\mathrm{Vect}(S^1)``, and its operator ``u\partial_x + \partial_x
u`` is the ``u``-dependent part of the **second KdV structure**. The square-root transformation
that makes it constant — see [Burgers](@ref) — is the dispersionless Miura map. The two
manuscripts are looking at the same object from two sides.

Two things are worth naming before going further, because they recur:

  - **The cancellation is special.** Under ``\bar{u} = \sqrt{u}`` the term that would spoil
    constancy is symmetric in ``A`` and ``B`` and drops out under antisymmetrisation. This is a
    feature of the one-dimensional transport bracket and does *not* happen for general
    Lie-Poisson brackets — `verify_burgers_leibniz.jl` shows it failing for vector fields in
    ``d > 1`` and for the canonical Vlasov bracket.
  - **Discretise the factorisation, not the operator.** That is the whole method, and the next
    two sections are what it buys.

## Hydrodynamic-type brackets, and prescribing the Casimir

The one-component Dubrovin-Novikov family factorises,

```math
\mathcal{J}_G = G(u)\partial_x + \tfrac12 G'(u) u_x = \sqrt{G} \,\cdot\, \partial_x \,\cdot\, \sqrt{G} ,
```

— multiplication, a *constant* antisymmetric operator, multiplication — and the discrete
counterpart is the same sandwich,

```math
\mathbb{J}_{ij}(u) = g_i(u_i) \, \mathbb{K}_{ij} \, g_j(u_j) , \qquad g_i = \sqrt{G_i} .
```

[`GaugedBracket`](@ref) is exactly this. It is Poisson for **any** constant antisymmetric
``\mathbb{K}`` and any smooth ``g_i`` of the **single** variable ``u_i``, because it is the
pushforward of ``\mathbb{K}`` under ``\bar{u}_i = \eta_i(u_i)`` with ``\eta_i' = 1/g_i``. In one
field component every metric ``G > 0`` is flat, which is why the continuum statement holds too.

Both hypotheses are sharp, and `verify_burgers_jacobi_family.jl` exhibits both failures rather
than asserting the theorem: let ``g_i`` depend on components other than its own, or let
``\mathbb{K}`` depend on ``u``, and the Jacobi identity goes.

The Casimirs are ``C_n(u) = \sum_i n_i \eta_i(u_i)`` for ``n \in \ker\mathbb{K}``, and since
``\operatorname{rank}\mathbb{J} = \operatorname{rank}\mathbb{K}`` these exhaust them.

### The Casimir is an input, not an output

Read ``\eta_i' = 1/g_i`` the other way round and the design changes character. One does not
choose ``g`` and then discover what is conserved; one **prescribes the Casimir density
``\eta``** and it determines ``g``:

```math
\eta'(u) = \frac{1}{g(u)} .
```

The flat coordinate is then the Casimir density itself, so ``C_n`` is *linear* there and every
Runge-Kutta method preserves it to round-off. Choosing ``\eta(u) = u\log u`` makes the
logarithmic entropy an exact Casimir by construction, with ``g(u) = 1/(\log u + 1)``.

!!! warning "The singularity is intrinsic"
    Wherever ``\eta'(u) = 0`` the gauge ``g`` blows up. For the entropy that is ``u = e^{-1}``,
    and no additive normalisation of ``\eta`` removes it — the location moves, the singularity
    does not. `verify_burgers_entropy_casimir.jl` §4 makes this precise.

    There is an escape, and it is instructive: for a Lie-Poisson bracket built from a matrix Lie
    algebra *every* spectral function ``\operatorname{tr}\eta(W)`` is a Casimir, so the
    logarithmic entropy comes free and without any singularity. That route needs (H) below, and
    is why the algebras of the next section are provided.

## Direct discretisation, and the condition it must meet

Discretising the bracket rather than its factorisation gives coefficients

```math
\mathbb{J}_{ij}(u) = \sum_m c_{ij}^m u_m , \qquad
c_{ij}^m = b_m \sum_{p,q} (\mathbb{M}^{-1})_{ip} \, [\varphi_p, \varphi_q](x_m) \, (\mathbb{M}^{-1})_{qj} ,
```

which is [`galerkin_c`](@ref). Such a bracket is Poisson if and only if the ``c_{ij}^m`` are the
structure constants of a Lie algebra — [`structure_constant_residual`](@ref). Two requirements
have to be kept apart, and conflating them is the mistake the manuscript exists to correct:

  - **(H) closure.** The ``c`` are the structure constants of *some* Lie algebra. This makes the
    Jacobi identity hold, exactly, at every ``N``.
  - **(C) asymptotic homomorphism.** The ``c`` converge to the continuum commutator. This makes
    the scheme consistent.

**Conventional finite elements deliver (C) and not (H), and (H) cannot be approximated.** The
condition is closed: there is no second-order-accurate Jacobi identity. Measured, the nodal
violation sits at ``\approx 0.73`` (``P_1``) and ``\approx 0.71`` (``P_2``) and does not move
under an eightfold refinement. The measurement is on the
[Diagnostics](@ref) page.

### The index convention

Throughout `src/algebras.jl` a three-tensor `C` holds ``c_{ij}^m`` at `C[m, i, j]`:

```math
[e_i, e_j] = \sum_m c_{ij}^m e_m , \qquad
\mathbb{J}_{ij}(z) = \sum_m c_{ij}^m z_m .
```

The leading index is what lets `C[m, :, :]` be a matrix and the whole tensor contract against a
coefficient vector with one [`lie_poisson_matrix`](@ref) call; it is also what
[`structure_constant_residual`](@ref) already expects. (The Python prototypes used
`c[i][j][m]`.) [`lie_poisson_derivative`](@ref) exists under its own name so that a caller
assembling a Jacobiator need not know that for a linear bracket it is `C` itself.

The **mode algebras** — [`witt_truncation`](@ref), [`poly_truncation`](@ref),
[`torus_truncation`](@ref) — return a named tuple `(; C, keep, con, labels)` splitting the
generators into a coarse block ``V_1`` = `keep` and a constrained block ``V_2`` = `con`, which
is what the [Dirac reduction](@ref) consumes.

## The algebras, and what each is for

| algebra | role |
|:--|:--|
| [`se3`](@ref) | **the positive control.** Semidirect rather than simple, and large enough that the Jacobi identity is a real constraint |
| [`random_antisymmetric_c`](@ref), ``n \ge 5`` | **the negative control** |
| [`so3`](@ref) | **a trap.** Bianchi class A — see the warning on the [Diagnostics](@ref) page. Provided so that the trap can be exhibited, not used |
| [`so_n`](@ref) | returns basis *and* constants, for the spectral Casimirs ``\operatorname{tr}\eta(W)`` |
| [`sine_algebra`](@ref), [`sine_coefficient`](@ref) | Zeitlin's ``\mathfrak{su}(N)`` truncation — the one discretisation here that closes **exactly** at finite ``N`` |
| [`witt_truncation`](@ref), [`graded_witt`](@ref), [`poly_truncation`](@ref), [`torus_truncation`](@ref) | mode truncations of ``\mathrm{Vect}(S^1)`` and of the torus algebra, split into ``V_1 \oplus V_2`` |

### Zeitlin's escape

The sine algebra is built from clock and shift matrices and has the closed form

```math
c_{mn}^{\,m+n} = \frac{N}{2\pi} \sin\!\left( \frac{2\pi}{N} \, m \times n \right) ,
\qquad m + n \bmod N , \quad N \text{ odd} ,
```

which closes into ``\mathfrak{su}(N)`` exactly and converges to the torus structure constants
``m \times n`` at order ``N^{-2}``:

```@example lp
using GeometricBrackets, Printf
for N in (11, 41, 161)
    _, c = sine_coefficient(N, (1, 0), (0, 1))
    @printf("N = %3d   c = %.6f   error = %.2e\n", N, c, abs(c - 1))
end
```

!!! note "Why sine_coefficient exists"
    [`sine_algebra`](@ref) returns the dense ``d^3`` tensor, which holds only ``d^2`` nonzeros:
    at ``N = 41`` that is 75 GB. [`sine_coefficient(N, m, n)`](@ref sine_coefficient) gives the
    single nonzero directly. Doing so also exposed that the ``N^{-2}`` law is *asymptotic* in
    ``N \gg 2\pi(m \times n)`` — at ``N = 11`` the observed order is 0.8, at ``N = 161`` it is
    1.98.

Zeitlin's construction works because its structure constants are **periodic in the grading**:
reducing an index mod ``N`` is invisible, since ``\sin`` is unchanged by ``m \times n \to m
\times n + N``. The Virasoro structure constants ``m - n`` are linear and unbounded and admit no
such repair; see [Aliasing and the zero-mode theorem](@ref).

A plain Fourier truncation — structure constants ``m \times n`` with indices wrapped mod ``N`` —
does **not** close, which is the control that shows the periodicity and not the wrapping is what
matters.

### Truncations close without being Lie algebras

```@example lp
w = witt_truncation(2)                    # |k| ≤ 2 kept, 2 < |k| ≤ 4 constrained
(closes = closes_on(w.C, w.keep, w.con),
 leak   = leak(w.C, w.keep, w.con),
 jacobi = structure_constant_residual(w.C))
```

``[V_1, V_1] \subseteq V_1 \oplus V_2`` exactly once ``q \ge 2``, because ``|k|, |l| \le K``
gives ``|k+l| \le 2K`` — and the truncated bracket is nevertheless not a Lie algebra, because
the pairs that leave the window were discarded. That gap is what [Dirac reduction](@ref) is
asked to close, and cannot.

## Four-brackets

!!! note "The continuum companion"
    Everything in this section is finite-dimensional and settled over ``\mathbb{Q}``. The same
    construction on the periodic two-torus — where the two-brackets reduce to
    ``\int u [A_u, B_u]`` exactly, and the weight that forces the reduction turns out to be
    singular — is [Poisson brackets from four-brackets](@ref).

Section 5 of the manuscript asks whether an antisymmetric **four**-bracket

```math
\{a,b;c,d\} = \sum \mathbb{K}^{ijkl} \, \partial_i a \, \partial_j b \, \partial_k c \, \partial_l d ,
\qquad \mathbb{K}^{ijkl} = -\mathbb{K}^{kjil} ,
```

can generate the two-brackets above, via ``\mathbb{J}_{ij} = \sum_{\alpha\beta} s_{1,\alpha}
\mathbb{K}^{i\alpha j\beta} s_{2,\beta}``. Two ansätze, with opposite verdicts.

**The Gardner-like ansatz** ``\mathbb{K}^{i\alpha j\beta} = \mu_{i\alpha}\sigma_{j\beta} -
\mu_{j\alpha}\sigma_{i\beta}`` gives a **decomposable** bivector ``\mathbb{J} = X \wedge Y``
with ``X = \mu\nabla s_1`` and ``Y = \sigma\nabla s_2``. Three identities collapse the whole
appendix, and none of them uses antisymmetry or diagonality of ``\mu, \sigma``:

  - ``\operatorname{rank}\mathbb{J} \le 2``, so the ansatz **cannot** reproduce the discrete
    Gardner operator ``\mathbb{M}^{-1}D\mathbb{M}^{-1}``, whose rank is ``N-1``;
  - ``\mathsf{P} - \mathsf{Q} = [X, Y]``, the Lie bracket of the two vector fields;
  - the 24-term reduced Jacobi condition equals exactly ``\det(X, \mathsf{P}-\mathsf{Q}, Y)``, so

    ```math
    \text{Jacobi} \iff X \wedge Y \wedge [X,Y] = 0 ,
    ```

    which is Frobenius involutivity — the classical criterion for a decomposable bivector.

**The Lie-Poisson four-bracket** ``\mathbb{K}^{i\alpha j\beta} = \delta^{\alpha\beta}
c_{ij}^{\alpha}`` with ``s = \tfrac23 \sum_i z_i^{3/2}`` (so that ``s_\alpha = \sqrt{z_\alpha}``)
gives ``\mathbb{J}_{ij} = \sum_m c_{ij}^m z_m``: **every** finite-dimensional Lie-Poisson
bracket, with Jacobi holding precisely when the ``c`` are structure constants. The ``3/2``
exponent is not a choice — it is forced within the separable ansatz ``s = \sum_\alpha
f_\alpha(z_\alpha)``. `verify_liepoisson_4bracket.jl` carries this end to end for
``\mathfrak{se}(3)``, and `verify_gardner_4bracket.jl` the identities above, both over
``\mathbb{Q}``.

## Exact arithmetic is part of the argument

Every structural claim on this page is an identity or its failure, never an approximation, so
the natural element type is `Rational{BigInt}`: over ``\mathbb{Q}``, "the identity holds" means
`iszero` rather than "the residual fell below a tolerance someone chose".

`LinearAlgebra`'s `rank` and `nullspace` route through the SVD and do not run there, and it is
exactly those two that the structural questions ask — *is there a Casimir, and how many*.
[`rref`](@ref), [`exact_rank`](@ref) and [`kernel`](@ref) answer them by Gauss-Jordan
elimination over any exact field. `det`, `inv`, `lu` and `\` already work on `Rational`, so
nothing else was needed.

The same code runs in `Float64` where a refinement study wants it. Keeping *one* implementation
rather than an exact copy and a floating-point twin is deliberate; the prototypes carried both
and recorded them drifting apart.

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["algebras.jl"]
```

### Exact linear algebra

```@autodocs
Modules = [GeometricBrackets]
Pages = ["exact.jl"]
```
