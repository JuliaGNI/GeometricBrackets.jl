#!/usr/bin/env julia
#
# The step-size sweep of the backward error analysis, as a figure.
#
#     julia --project=scripts scripts/kdv_bea_sweep.jl [--outdir=PATH]
#
# The one figure of the KdV manuscript that `scripts/kdv.jl` does not produce. Two panels: the
# departure of the respective other Hamiltonian against dt, and its excess over the dt -> 0
# plateau. Writes `kdv-bea-dtsweep.pdf`.
#
# The excess is what carries the dt dependence, and on the upper panel it is invisible -- the
# midpoint curve on the first flow is flat and the others sit within a factor of eight of it.
# Subtracting each run's own plateau, taken as its value at the smallest step size, leaves the
# O(dt^2) of Propositions 8.7 and 8.8, and leaves the midpoint rule on the first flow with
# nothing at all, which is Theorem 8.6.
#
# The numbers are the ones `verify_kdv_bea.jl` section 6 checks; this script only draws them.

using GeometricBrackets
using CairoMakie
using Printf

include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

const SWEEP_N, SWEEP_P, SWEEP_T = 16, 3, 10.0
const SWEEP_DTS = (3.2e-3, 1.6e-3, 8e-4, 4e-4, 2e-4)

# (label, stepper, which invariant to watch: 1 = H1, 2 = H2)
const SWEEP_RUNS = (
    ("midpoint, flow 1", (s, d, dt) -> midpoint(s, d, dt, 1)[1], 2),
    ("AVF, flow 1", (s, d, dt) -> avf(s, d, dt, 1)[1], 2),
    ("discrete gradient, flow 1", (s, d, dt) -> dgrad(s, d, dt), 2),
    # the package's palette key is "discrete gradient (mass)" -- a lighter shade of its
    # Euclidean counterpart, so the pair reads as one method with two metrics. The
    # Python spelled it "discrete gradient, M", which `_series_style` would parse to the
    # plain key and draw the two identically.
    ("discrete gradient (mass), flow 1", (s, d, dt) -> dgrad_mass(s, d, dt), 2),
    ("midpoint, flow 2", (s, d, dt) -> midpoint(s, d, dt, 2)[1], 1)
)

function outdir_from(args)
    for a in args
        startswith(a, "--outdir=") && return a[10:end]
    end
    joinpath(@__DIR__, "figures")
end

function sweep_case()
    s = KdVSys(SWEEP_N; p = SWEEP_P, u0 = x -> 0.1 * sin(x))
    @printf("  step-size sweep: N = %d, p = %d, u0 = 0.1 sin x, T = %g\n",
        SWEEP_N, SWEEP_P, SWEEP_T)
    out = Pair{String, Vector{Float64}}[]
    for (name, step, q) in SWEEP_RUNS
        row = Float64[]
        for dt in SWEEP_DTS
            NT = round(Int, SWEEP_T / dt)
            _, ref, dev = integrate_windowed(s, d -> step(s, d, dt), NT;
                stride = max(1, NT ÷ 200))
            push!(row, env_drift(dev[q], ref[q]))
        end
        push!(out, name => row)
        @printf("      %-30s %s\n", name, join([@sprintf("%.3e", v) for v in row], "  "))
    end
    out
end

function main(args)
    out = outdir_from(args)
    mkpath(out)
    series = sweep_case()
    fig = sweepplot(collect(SWEEP_DTS), series;
        title = "u₀ = 0.1 sin x, N = $SWEEP_N, p = $SWEEP_P:  " *
                "maxₜ |H(t) − H(0)| / |H(0)|,  t ≤ $(Int(SWEEP_T))\n" *
                "H₂,d along the first flow, H₁,d along the second")
    path = joinpath(out, "kdv-bea-dtsweep.pdf")
    save(path, fig)
    println("      wrote $path")
    return 0
end

exit(main(ARGS))
