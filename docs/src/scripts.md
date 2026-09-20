```@meta
CurrentModule = GeometricBrackets
```

# The verification scripts

`scripts/` holds twenty-seven `verify_*.jl` scripts that machine-verify the claims of the three
manuscripts, two that draw their figures, one exploratory search, a driver and four shared
helper modules — thirty-five files, plus the seven in `fable/`. Eighteen of the verification
scripts are converted Python prototypes that used to live in the manuscripts' `Scripts/`
directories; [Verification](@ref) records the agreement claim by claim and the errata the
comparison turned up. The four `verify_fourbracket_*` scripts and the five of the metric-bracket
half had no Python ancestor — they arrived as standalone Julia, or alongside the Julia
implementation itself.

This page is an **index**: what each script checks, and where the theory behind it is written
down. Each script's own header carries the authoritative section-by-section detail, which is
why the entries here are one line long.

## Running them

```sh
julia --project=scripts scripts/run_all.jl          # all twenty-seven; nonzero exit on failure
julia --project=scripts scripts/verify_kdv_bea.jl   # or just one
sh scripts/run_fable.sh                             # the seven fable scripts, about 13 minutes
                                                    # (longer the first time: one of them needs SymPy)
```

Each script runs in its own process, so a failure — or an `exit(1)` from the harness — is
contained and reported rather than taking the runner down. `search_dirac_variants.jl` is
exploratory and slow and is deliberately excluded; run it by hand.

The seven `fable/` scripts have a driver of their own rather than a place in `run_all.jl`. They
are verification scripts on the same terms as the rest — not exploratory — but
`fable/III4_multifield_derivative.jl` takes about eleven minutes by itself, which is longer than
all of `run_all.jl` together, so folding them in would change what that driver is for.

The symbolic scripts need SymPy, which `CondaPkg` provisions on first use through
`SymPyPythonCall`. **That dependency is confined to `scripts/Project.toml`**: the package, its
test suite and its CI acquire no Python.

### The harness

`scripts/check.jl` is the PASS/FAIL harness — `header`, `check`, `summary`. Its output format is
copied deliberately from the `common.py` it replaces, down to the two leading spaces and the
three before the detail, so that a converted script could be checked against its original with a
plain `diff` rather than by reading two tables side by side.

Every `check` carries a **detail**: the residual, rate or rank behind the verdict. A bare PASS
says only that someone chose the tolerance well; the number says what was actually seen, and it
is the number that was compared.

`scripts/rationals.jl` supplies random exact rationals (`rnd`, `rnd_vec`, `rnd_mat`), matching
the Python's ranges so residuals land in the same ballpark — not the same numbers, since the two
random streams differ, and no attempt was made to make them agree. Where a claim can be settled
on deterministic data instead — ``\mathfrak{se}(3)``, ``\mathfrak{so}(N)``, a fixed mesh — that
is preferred, and those *do* agree exactly.

Two idioms sit alongside `check` for claims about a continuum identity rather than a matrix.
`check_exact` settles one against an absolute threshold, for an identity that is algebra plus
integration by parts on band-limited fields and must therefore land at roundoff. `check_refined`
settles one against the factor by which the residual drops when the resolution is doubled, for an
identity involving a quotient, a power or a logarithm, where no absolute threshold applies
because the residual is discretisation error. It prints both residuals and the factor —
`3.14e-08 -> 1.25e-10   251x` — because with a predicted 256 the factor *is* the evidence;
a bare PASS would say only that someone chose the threshold well. Neither knows anything about
grids: which resolutions a refinement uses is the calling manuscript's business.

`scripts/torustools.jl` holds that business for the four-bracket manuscript — the fixed test
fields, and the two- and four-grid refinement drivers. The fields were carried in three copies
before it existed, and one of them in a fourth as a local variable.

`scripts/fable/fabletools.jl` is the equivalent for the fable scripts: trigonometric polynomials
over ``\mathbb{Q}(i)`` on ``T^D``, first-order jets for Gateaux derivatives, and the canonical
bracket on ``T^{2n}``. Plain definitions rather than a module, so a script can extend `∂`, `mean`
and `phasedim` to its own field type — which `III1_structure_function_4d.jl` does for its
floating-point grids. The two `*_refutation.jl` scripts deliberately use **none** of it, and
nothing from the package either: they import `LinearAlgebra` and `Test` and `include` nothing, so
that a defect in the shared machinery cannot hide in both the claim and its check.

`scripts/kdvtools.jl` is the shared KdV assembly. The Python carried `nq_for` in four files, a
`Sys` class in three and the ``\mathbb{K}^2`` blocks in two; [Verification](@ref) records what
that duplication cost when a sign convention changed and only some copies moved. One definition
each, here.

## The KdV manuscript

| script | what it establishes | theory |
|:--|:--|:--|
| `verify_kdv_continuous.jl` | the continuous bi-Hamiltonian structure in jet variables: the two variational derivatives, that both structures generate the same equation, skew-adjointness of ``D_2``, and ``\{H_2,H_1\}_1 = \{H_1,H_2\}_2 = 0`` | [Korteweg-de Vries](@ref) |
| `verify_kdv_discrete.jl` | the discrete brackets on a periodic B-spline basis: ``S`` already antisymmetric, the third-derivative block, ``\mathbb{P}^1`` exactly Poisson and ``\mathbb{P}^2`` not — and that cross-conservation on a uniform mesh is circulance, not an identity in the degrees of freedom | [Discretisation](@ref), [Diagnostics](@ref) |
| `verify_kdv_semidiscrete.jl` | what each bracket generates: a mixed two-field Galerkin method and the plain Galerkin scheme, differing at ``O(h^{2p})``; conservation of each Hamiltonian and of the mass; Magri's first rung exact and the second only ``O(h^{2p})`` | [Korteweg-de Vries](@ref) |
| `verify_kdv_timedisc.jl` | which invariants survive time discretisation: the mass is free, midpoint is a Poisson map, Ge-Marsden, Cooper's hypothesis, and projection onto both level sets | [Integrators](@ref), [Diagnostics](@ref) |
| `verify_kdv_jacobi_family.jl` | the integration-by-parts family ``\alpha T^1 + \beta T^3`` has one relation, antisymmetry forces ``\alpha = 2\beta``, and **no** member closes into a Lie algebra | [Diagnostics](@ref) |
| `verify_kdv_aliasing.jl` | Zeitlin-style aliasing cannot repair the second bracket — the zero-mode theorem, and the cocycle obstruction | [Aliasing and the zero-mode theorem](@ref) |
| `verify_kdv_miura.jl` | the Miura chart: the factorisation, where the discrete map is a diffeomorphism, exact Poissonness at every ``N``, where the chart lives, the spectral parameter, and that the two charts are not the same numerical method | [The Miura map](@ref) |
| `verify_kdv_nambu.jl` | the fully antisymmetric three-bracket question, end to end | [Antisymmetric three-brackets](@ref) |
| `verify_kdv_bea.jl` | why implicit midpoint holds ``H_2`` so well: the symmetry group, the modified Hamiltonian, the exact increment lemma, the step-size sweep and the ``2p+2`` exponent | [Backward error analysis](@ref) |

## The Lie-Poisson manuscript and its Dirac companion

| script | what it establishes | theory |
|:--|:--|:--|
| `verify_burgers_jacobi_family.jl` | ``\mathbb{J}_{ij} = g_i(u_i)\mathbb{K}_{ij}g_j(u_j)`` is Poisson, **both** hypotheses sharp, and its Casimirs | [Burgers](@ref), [Discrete Lie-Poisson brackets](@ref) |
| `verify_burgers_leibniz.jl` | where the square-root transformation constantifies a Lie-Poisson bracket, and where it does not — the four-term expansion is false, ``[hF,hG] = h^2[F,G]`` is what is needed, and it fails in ``d>1`` and for the Vlasov bracket | [Discrete Lie-Poisson brackets](@ref) |
| `verify_burgers_entropy_casimir.jl` | prescribing the Casimir, ``\eta' = 1/g``; the continuum factorisation; the intrinsic singularity; and the spectral-Casimir alternative | [Discrete Lie-Poisson brackets](@ref) |
| `verify_burgers_discretisation.jl` | the discrete Burgers bracket: ``\mathbb{K}`` antisymmetric of rank ``N-1``, the exact ``\sqrt{u}`` Casimir, second-order consistency, and integration in ``\bar{u}`` | [Burgers](@ref) |
| `verify_liepoisson_structure_constants.jl` | nodal finite elements violate the structure-constant condition and refinement does not help; the sine bracket closes into ``\mathfrak{su}(N)`` exactly; a plain Fourier truncation does not | [Discrete Lie-Poisson brackets](@ref) |
| `verify_liepoisson_4bracket.jl` | the antisymmetric four-bracket generates **every** finite-dimensional Lie-Poisson bracket, and the ``3/2`` power is forced | [Four-brackets](@ref) |
| `verify_zeitlin_three_bracket.jl` | the Bialynicki-Birula--Morrison three-bracket ``[A,B,I_2]`` reproduces the Zeitlin bracket on ``\mathfrak{su}(N)``, the truncation supplying the non-degenerate Killing form the continuum lacks; with ``\mathfrak{se}(3)`` and the cubic Casimir as the controls that fail | [Discrete Lie-Poisson brackets](@ref) |
| `verify_gardner_4bracket.jl` | the Gardner-like ansatz is decomposable, hence rank ``\le 2``; Jacobi ``\iff`` Frobenius involutivity | [Four-brackets](@ref) |
| `verify_dirac_reduction.jl` | the Jacobiator transports tensorially through the Dirac projector, so reduction preserves the Jacobi identity and never creates it; the four realisations; what survives | [Dirac reduction](@ref) |
| `search_dirac_variants.jl` | an exhaustive search for a variant that *does* create it — B1a–B4. **Exploratory and slow; excluded from `run_all.jl`** | [The search for a repair](@ref) |

## The four-bracket manuscript

The continuum companion to the four-bracket sections above: fields on the periodic two-torus
rather than structure constants over ``\mathbb{Q}``.

| script | what it establishes | theory |
|:--|:--|:--|
| `verify_fourbracket_metriplectic.jl` | Proposition 2.1: the two-bracket induced by a Kulkarni-Nomizu product of positive semi-definite tensors is positive semi-definite, at every rank pair — and that relaxing semi-definiteness breaks it, so the proposition is not vacuous | [Poisson brackets from four-brackets](@ref) |
| `verify_fourbracket_identities.jl` | every displayed identity of Sections 3 to 6: the Gardner auxiliary identities, both reductions to ``\int u[A_u,B_u]`` and the factor of two between them, the Plücker relation, Theorem 4.5 for five unrelated structure functions, the Casimirs, the weighted brackets, Section 6's vanishing bracket, and Lemma 3.1 — that each family satisfies exactly **one** of the two antisymmetry conditions | [Poisson brackets from four-brackets](@ref) |
| `verify_fourbracket_convergence.jl` | that the six identities involving non-band-limited fields converge at the design order of the stencil, so their residuals are discretisation error and not a defect of the identity; observed orders 7.2 to 8.1 | [Poisson brackets from four-brackets](@ref) |
| `verify_fourbracket_log_entropy.jl` | the log-entropy weight: the pole at ``u = e^{-1}`` cancels in the two-bracket and diverges in the four-bracket, no regular weight can replace it, and the Casimir shift moves it out of range | [Poisson brackets from four-brackets](@ref) |

## The metric-bracket half

The symmetric, dissipative half of a metriplectic structure, and the two-dimensional spaces it
needs. None of these scripts had a Python ancestor.

| script | what it establishes | theory |
|:--|:--|:--|
| `verify_metric_collapse.jl` | that the collision-like bracket's ``O(N_q^2)`` double integral collapses to fourteen global moments **exactly**: the identity ``Q_2(z) = z^\perp \otimes z^\perp`` over five decades of scale, sufficiency and minimality of the moment set, and the three hypotheses whose failure breaks it — a non-polynomial kernel, a ``\kappa`` that does not factorise, an ``x``-dependent measure. Recentring is established as a *conditioning* requirement, swept over draws | [`CollisionBracket`](@ref) |
| `verify_metriplectic_flow.jl` | that ``H`` is conserved by the bracket's degeneracy rather than by the integrator and ``S`` falls by semi-definiteness alone; `metric_directional` against the ``O(N^3)`` tensor contracted by hand, with the speedup ratios for all four brackets; and why Newton must be run to a residual tolerance rather than to a fixed iteration count | [`MetriplecticFlow`](@ref) |
| `verify_pullback.jl` | that the pullback of a mapped domain's measure and metric is the right one: on an annulus, where ``\lvert\det J\rvert = r`` and ``\mathbb{D} = \mathrm{diag}(r, 1/r)`` in closed form, and the weak Laplacian ``\int\nabla u\cdot\nabla v \,d\mu = -\int v\Delta u \,d\mu`` converges under refinement while a plain stiffness and a dropped metric each sit at their own wrong constant | [`PulledBack`](@ref) |
| `verify_polar_bracket.jl` | that a bracket survives the pole: ``\int\lvert\nabla u\rvert^2 dx = 2\pi`` on the whole unit disk, a question no tensor-product space can be asked, and that symmetry, semi-definiteness and degeneracy all hold on the polar space — with a wrong generator, a sign-changing mobility and a ``z\otimes z`` kernel as the controls that break exactly one property each | [`PolarSplineSpace`](@ref) |
| `verify_frame_covariance.jl` | that [`CollisionBracket`](@ref) reads the **physical** frame on a mapped domain: against the ``O(N_q^2)`` double sum of its own definition, with the physical gradient written out from the polar chart by hand, on an annulus and again on the `PolarSplineSpace` through the pole; and against itself under a reparametrisation. The frame dropped and the frame transposed are the two controls, each an ``O(1)`` different operator — and symmetry, semi-definiteness and degeneracy hold for **both** of them, which is why covariance is a separate check. It also measures how far the reparametrisation check reaches: a linear family has a constant ``J``, and for a constant ``J`` the bracket depends on the frame only through ``\lvert\det J\rvert``, so a transposed frame passes it exactly | [`MappedFrame`](@ref) |

## The fable investigation

Seven scripts from an autonomous investigation into discrete Poisson brackets. Five establish
results; three of those are now backed in the manuscripts, and two further scripts are adversarial
re-checks written to *overturn* an earlier one and sharing no code with it. Run by
`scripts/run_fable.sh`, not by `run_all.jl`. `fable/II1_semidirect_flat.jl` is the one symbolic
script among them, so a first run also pays for the SymPy provisioning described above.

| script | what it establishes | theory |
|:--|:--|:--|
| `fable/I1_vlasov_uN.jl` | finite-mode Vlasov–Poisson on a bounded domain is exactly Lie-Poisson on ``\mathfrak{u}(N)^*``; the homomorphism defect vanishes on the inner block for symbols of degree ``\le 1``, and at degree 2 is the Wigner correction | [Discrete Lie-Poisson brackets](@ref) |
| `fable/II1_semidirect_flat.jl` | semidirect-product brackets (1-D Euler, shallow water) as pushforwards of a constant bracket under a pointwise chart; entropy-carrying systems are **not** in the class | [Discrete Lie-Poisson brackets](@ref) |
| `fable/III1_structure_function_4d.jl` | ``\int c(u)\{A_u,B_u\}\mu`` is Poisson for every smooth ``c`` in every dimension, the obstruction being exact rather than pointwise zero | [Poisson brackets from four-brackets](@ref) |
| `fable/III1_refutation.jl` | an independent re-derivation of the same statement that fails to overturn it, and finds the sign of the top-form lemma to be Liouville-specific; the non-Liouville control is non-zero | [Poisson brackets from four-brackets](@ref) |
| `fable/III2_matrix_structure_function.jl` | the matrix analogue ``\operatorname{tr} c(W)[F_W,G_W]`` is Poisson **iff** ``c`` is affine for ``N \ge 3``; the spectral-kernel classification and the log-mean pushforward | [Poisson brackets from four-brackets](@ref) |
| `fable/III2_refutation.jl` | the same, re-checked by brute-force eigenvector perturbation — the one step no closed form reaches; and that the pushforward family needs a definite spectrum | [Poisson brackets from four-brackets](@ref) |
| `fable/III4_multifield_derivative.jl` | several fields: Poisson iff the induced product is associative; and structure functions may never depend on ``\nabla f``. **About eleven minutes** | [Poisson brackets from four-brackets](@ref) |

## The figures

Both take `--outdir=PATH`, which is how the manuscripts' own `figures/` directories are
regenerated.

```sh
julia --project=scripts scripts/kdv.jl                # all six cases
julia --project=scripts scripts/kdv.jl cos            # or just one
julia --project=scripts scripts/kdv_bea_sweep.jl
```

`kdv.jl` runs the cases `cos`, `soliton1`, `soliton2`, `soliton3`, `soliton5` and `miura`, and
writes per case:

| file | contents |
|:--|:--|
| `kdv-<case>-{H1,H2,C0}-flow1.pdf` | the four flow-1 runs |
| `kdv-<case>-{H1,H2,C0}-other.pdf` | the two flow-2 runs, and the two Miura runs |
| `kdv-<case>-state-{flow1,other}.pdf` | initial and final states, split the same way |
| `<case>.md` | the maximum error in each invariant, with the growth of its envelope |

`kdv_bea_sweep.jl` writes `kdv-bea-dtsweep.pdf`, the one figure `kdv.jl` does not produce. See
[Reading the sweep figure](@ref).

Why the figures are split the way they are — one invariant per figure, one family of vector
fields per figure, colour for the integrator and line style for the flow — is on the
[Diagnostics](@ref) page under [Plotting](@ref).

## Where the library ends and a script begins

The rule is that a script **diagnoses what the package ships** rather than a copy of it.
`verify_burgers_discretisation.jl` uses [`LagrangeSpace`](@ref), [`burgers_bracket`](@ref) and
[`burgers_casimir`](@ref); `verify_kdv_miura.jl` uses [`miura_map`](@ref),
[`miura_invert`](@ref), [`hill_lambda0`](@ref) and [`kdv_miura_bracket`](@ref);
`verify_dirac_reduction.jl` uses `src/dirac.jl` and `src/hierarchical.jl` unchanged; the four
`verify_fourbracket_*` scripts use [`TorusGrid`](@ref), every bracket in `src/fourbrackets.jl`
and [`metriplectic_bracket`](@ref), and define between them one helper apiece — a random
positive semi-definite matrix, and the entropy ``s_\alpha = u\log u + \alpha u`` of
Example 5.4, which is one example rather than a family the package should ship.

The exceptions are deliberate and are flagged in the scripts that make them. Sections 1 and 3 of
`verify_burgers_jacobi_family.jl` write out ``\mathbb{J}_{ij} = g_i\mathbb{K}_{ij}g_j`` by hand
rather than taking [`GaugedBracket`](@ref), because the point of those sections is to let ``g_i``
vary with the index and then depend on components other than its own — which the type, carrying
a single scalar `g` applied componentwise, cannot express.
