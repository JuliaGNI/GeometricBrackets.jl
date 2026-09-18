#!/usr/bin/env julia
#
# Run every verification script.  Exits nonzero if any check fails.
#
#     julia --project=scripts scripts/run_all.jl [script ...]
#
# With no arguments it runs the whole suite, in the order below: the KdV scripts follow the
# sections of `discrete-kdv-brackets-notes.tex`, the Burgers and Lie-Poisson ones those of
# `discrete-lie-poisson-brackets.tex` and its Dirac companion, the four-bracket ones those of
# `poisson-brackets-from-four-brackets.tex`.  Naming each script after the bracket it diagnoses
# is what keeps the three manuscripts apart in one flat directory.
#
# Each script runs in its own process, so a failure -- or an `exit(1)` from `summary` -- is
# contained and reported rather than taking the runner down with it.  The symbolic scripts
# need SymPy, which CondaPkg provisions on first use; the rest are pure Julia.
#
# `search_dirac_variants.jl` is exploratory and slow, and is deliberately excluded, exactly
# as it is from the `run_all.sh` it replaces.  Run it by hand.
#
# The seven scripts under `fable/` are also excluded, and for a different reason: they are
# verification scripts on the same footing as these, but `fable/III4_multifield_derivative.jl`
# alone takes about eleven minutes.  They have their own driver, `scripts/run_fable.sh`.

const SCRIPTS = [
    # the KdV bracket pair: P1 constant and Poisson, P2 antisymmetric but not Jacobi
    "verify_kdv_continuous.jl",
    "verify_kdv_discrete.jl",
    "verify_kdv_semidiscrete.jl",
    "verify_kdv_timedisc.jl",
    "verify_kdv_jacobi_family.jl",
    "verify_kdv_aliasing.jl",
    "verify_kdv_miura.jl",
    "verify_kdv_nambu.jl",
    "verify_kdv_bea.jl",
    # the Burgers bracket and the discrete Lie-Poisson family around it
    "verify_burgers_jacobi_family.jl",
    "verify_burgers_leibniz.jl",
    "verify_burgers_entropy_casimir.jl",
    "verify_burgers_discretisation.jl",
    "verify_liepoisson_structure_constants.jl",
    "verify_liepoisson_4bracket.jl",
    "verify_zeitlin_three_bracket.jl",
    "verify_gardner_4bracket.jl",
    "verify_dirac_reduction.jl",
    # the continuum four-brackets, and which of them reduce to a Lie-Poisson bracket
    "verify_fourbracket_metriplectic.jl",
    "verify_fourbracket_identities.jl",
    "verify_fourbracket_convergence.jl",
    "verify_fourbracket_log_entropy.jl",
    # the metric half: the collision-like bracket's O(N_q) moment collapse, and the flow the
    # metric brackets generate once a Hamiltonian and an entropy are attached to them
    "verify_metric_collapse.jl",
    "verify_metriplectic_flow.jl",
    # the mapped domain: the pullback of a measure and a metric, and the polar space that
    # carries a bracket across the pole the tensor-product spaces cannot reach
    "verify_pullback.jl",
    "verify_polar_bracket.jl",
    "verify_frame_covariance.jl"
]

const RULE = "="^67

function main(args)
    requested = isempty(args) ? SCRIPTS : args
    project = @__DIR__
    failed = String[]

    for s in requested
        path = joinpath(@__DIR__, s)
        isfile(path) ||
            (push!(failed, "$s (missing)"); @error "no such script" script=s; continue)
        println(RULE)
        println(s)
        println(RULE)
        flush(stdout)
        cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no $path`
        success(pipeline(cmd; stdout, stderr)) || push!(failed, s)
    end

    println()
    if isempty(failed)
        println("All verification scripts passed.")
        println()
        println("search_dirac_variants.jl is exploratory and slow; run it by hand.")
        return 0
    end
    println("$(length(failed)) SCRIPT(S) FAILED: " * join(failed, ", "))
    return 1
end

exit(main(ARGS))
