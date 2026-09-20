#!/usr/bin/env julia
#
# The numerical experiments of the KdV manuscript: the relative error of BOTH Hamiltonians
# against time, and the initial and final states, for each of the standard cases.
#
#     julia --project=scripts scripts/kdv.jl [case ...]
#
# with cases `cos`, `soliton1`, `soliton2`, `soliton3`, `soliton5` and `miura`. With no
# arguments it runs all six. Per case it writes into scripts/figures/:
#
#   kdv-<case>-{H1,H2,C0}-flow1.pdf   the four flow-1 runs
#   kdv-<case>-{H1,H2,C0}-other.pdf   the two flow-2 runs, and the two Miura runs where
#                                     the case has them
#   kdv-<case>-state-{flow1,other}.pdf   initial and final states, split the same way
#   <case>.md                         a table of the maximum error in each invariant
#
# In every case each geometric method holds the Hamiltonian generating its flow to round-off
# and lets the other drift by orders of magnitude more. RK4's error is the only one that
# genuinely grows, and on the narrow examples it also pays for the structure it lacks in
# step count.
#
# The `miura` case is different in kind from the other five and adds a THIRD vector field.
# It is posed in v, not in u, and it has to be: int u_h = -int v_h^2, so the image of the
# discrete Miura map lies in C_0,d <= 0, and none of the other five -- cos x has zero mass,
# the solitons are elevations of positive mass -- has a preimage at all.

using CairoMakie
using GeometricBrackets

"""
Where the figures go. Defaults to `scripts/figures/`; `--outdir=PATH` redirects it, which is
how the manuscript's own `Scripts/figures/` is regenerated now that `plot_kdv_energies.py`
is gone.
"""
const FIGDIR = let d = joinpath(@__DIR__, "figures")
    for a in ARGS
        startswith(a, "--outdir=") && (d = a[10:end])
    end
    d
end

const T_END = 100.0          # the four small examples run to the same final time
const RK4_SAFETY = 0.8       # fraction of the RK4 stability limit actually used
const NOUT = 400             # output windows per run

Base.@kwdef struct Case
    key::String
    title::String
    n::Int
    p::Int = 3
    L::Float64
    T::Float64
    dt::Float64
    u0 = nothing             # posed in u ...
    v0 = nothing             # ... or, for the Miura case, in v
    xleft::Float64 = 0.0     # only the plots carry the box back to its own coordinates
    note::String = ""
end

const LSOL = 40.0

const CASES = [
    Case(key = "cos", title = "u₀ = cos x",
        n = 64, L = 2π, T = T_END, dt = 1e-3,
        u0 = cosine(2π)),
    Case(key = "soliton1", title = "single soliton",
        n = 128, L = LSOL, T = T_END, dt = 5e-3,
        u0 = soliton(1.0, 10.0, LSOL), note = "ten full traversals"),
    Case(key = "soliton2", title = "two-soliton",
        n = 128, L = LSOL, T = T_END, dt = 5e-3,
        u0 = two_soliton(1.2, 0.8, 12.0, 22.0, LSOL), note = "eight overtakings"),

    # Shi, Fu and Liu, Appl. Math. Comput. 508 (2026) 129620, Problem 4.2
    Case(key = "soliton3", title = "three solitons",
        n = 256, L = 200.0, T = 250.0, dt = 0.1,
        u0 = three_solitons(), xleft = -100.0,
        note = "the tall wave overtakes both"),

    # ... and Problem 4.3
    Case(key = "soliton5", title = "five solitons",
        n = 384, L = 300.0, T = 500.0, dt = 1/16,
        u0 = five_solitons(), xleft = -150.0,
        note = "the tall wave overtakes all four"),

    # posed in v; u₀ = M_h(v₀)
    Case(key = "miura", title = "Miura, v₀ = 1 + sin x",
        n = 64, L = 2π, T = T_END, dt = 1e-3,
        v0 = miura_initial_v(2π), note = "u₀ = M_h(v₀)")
]

function find_case(k)
    (i = findfirst(c -> c.key == k, CASES);
        i === nothing ?
        error("unknown case $(k); known: " *
              join((c.key for c in CASES), ", ")) : CASES[i])
end

"""The six runs posed in u: colour carries the integrator, line style the vector field."""
function runs_u(sys)
    (("midpoint, flow 1", sys.flow1, ImplicitMidpoint(), false),
        ("midpoint, flow 2", sys.flow2, ImplicitMidpoint(), false),
        ("discrete gradient, flow 1", sys.flow1, Gonzalez(), false),
        ("AVF, flow 1", sys.flow1, AverageVectorField(), false),
        ("RK4, flow 1", sys.flow1, RungeKutta4(), true),
        ("RK4, flow 2", sys.flow2, RungeKutta4(), true))
end

"""The two Miura runs, carried in v. They cannot be added to `runs_u`: a MiuraSystem holds
v̂ as its state, so the two families never appear on the same system object — only on the
same figure, through the same initial u_h."""
function runs_v(sysM)
    (("midpoint, Miura", sysM.flow, ImplicitMidpoint(), false),
        ("discrete gradient, Miura", sysM.flow, Gonzalez(), false))
end

function run_one(sys, flow, meth, x0, dt, T)
    NT = max(round(Int, T / dt), NOUT)
    # `round`, not `div`, so that the windowing matches `plot_kdv_energies.py` step for step;
    # `integrate` takes the remainder in a final short window either way.
    stride = max(round(Int, NT / NOUT), 1)
    # û₀ is passed so that the Newton tolerance scales with the amplitude of the field; the
    # Miura initial condition reaches |u| ~ 4 where the solitons are below 1, and one
    # absolute tolerance cannot serve both
    traj = integrate(sys, Integrator(flow, meth, dt; û₀ = x0), x0, NT; stride = stride)
    (traj, NT)
end

"""Log and flush.

`@info` writes to a buffered stream, and a run that is killed part-way — these take tens of
minutes — then leaves an empty log and no indication of how far it got. Flushing after every
line costs nothing at this rate and makes the log worth reading.
"""
function note(msg)
    println(stderr, "[ ", msg)
    flush(stderr)
end

function report(label, traj)
    note(
        "  " * rpad(label, 26) *
        " |ΔH₁| = $(round(drift(traj, :H1), sigdigits = 2))" *
        "  |ΔH₂| = $(round(drift(traj, :H2), sigdigits = 2))" *
        "  |ΔC₀| = $(round(absolute_drift(traj, :C0), sigdigits = 2))" *
        "  g(H₁) = $(round(growth(traj, :H1), sigdigits = 2))")
end

"""The two families the figures are split into.

`flow 1` is the mixed two-field method, `flow 2` the plain Galerkin one, and `Miura` the
mKdV chart. Keeping the first family on its own axis is what makes the comparison legible:
its four runs differ by ten orders of magnitude in `H₁`, which is the point, and putting the
other family beside them on a shared axis hides it."""
group_of(label) = occursin("flow 1", label) ? :flow1 : :other

const GROUP_TITLE = Dict(:flow1 => "flow 1", :other => "flow 2 and Miura")

"""Rows of the summary table, in the order the runs were made."""
struct RunSummary
    label::String
    dt::Float64
    steps::Int
    dH1::Float64
    dH2::Float64
    dC0::Float64
    gH1::Float64
    gH2::Float64
end

function summarise(label, traj, dt, steps)
    RunSummary(label, dt, steps,
        drift(traj, :H1), drift(traj, :H2), absolute_drift(traj, :C0),
        growth(traj, :H1), growth(traj, :H2))
end

"""Two significant digits, which is all any of these numbers carries."""
fmt(x) = x == 0 ? "0" : string(round(x; sigdigits = 2))

function write_summary(case::Case, rows::Vector{RunSummary}, s)
    path = joinpath(FIGDIR, "$(case.key).md")
    open(path, "w") do io
        println(io, "# ", case.title)
        println(io)
        isempty(case.note) || (println(io, case.note); println(io))
        println(io, "## Setup")
        println(io)
        println(io, "| | |")
        println(io, "|:--|:--|")
        println(io, "| domain | ``[", round(case.xleft; digits = 1), ", ",
            round(case.xleft + case.L; digits = 1), ")`` |")
        println(io, "| degrees of freedom | ", nbasis(s), " |")
        println(io, "| spline degree | ", case.p, " |")
        println(io, "| final time | ", case.T, " |")
        println(io, "| step size (implicit) | ", case.dt, " |")
        println(io, "| posed in | ", case.v0 === nothing ? "``u``" : "``v``", " |")
        println(io)

        println(io, "## Invariant errors")
        println(io)
        println(io, "Maximum over the run of the deviation from the initial value. ",
            "``H_1`` and ``H_2`` are relative, ``C_0`` absolute — the mass is zero ",
            "for some of these initial conditions, and a relative error would be ",
            "meaningless there.")
        println(io)
        println(
            io, "The last two columns are the *growth* of the error envelope, the ratio ",
            "of its size over the last tenth of the run to the first. Near one means ",
            "a bounded oscillation; large means a secular drift. This is the number ",
            "that separates \"the invariant departs and stays there\" from \"the ",
            "invariant is being lost\", and a single end-of-run figure cannot tell ",
            "them apart.")
        println(io)
        println(io, "| run | Δt | steps | max ΔH₁ | max ΔH₂ | max ΔC₀ | growth H₁ | growth H₂ |")
        println(io, "|:--|--:|--:|--:|--:|--:|--:|--:|")
        for r in rows
            println(io, "| ", r.label, " | ", fmt(r.dt), " | ", r.steps, " | ",
                fmt(r.dH1), " | ", fmt(r.dH2), " | ", fmt(r.dC0), " | ",
                fmt(r.gH1), " | ", fmt(r.gH2), " |")
        end
        println(io)

        # The reading, stated rather than left to the reader to reconstruct. Each claim is
        # computed from the table above rather than asserted, so that a run which behaves
        # differently from the expected pattern says so instead of being papered over.
        best1 = rows[argmin([r.dH1 for r in rows])]
        best2 = rows[argmin([r.dH2 for r in rows])]
        uchart = filter(r -> !occursin("Miura", r.label), rows)
        vchart = filter(r -> occursin("Miura", r.label), rows)

        println(io, "## Reading")
        println(io)
        println(
            io, "- ``H_1`` is held best by **", best1.label, "** (", fmt(best1.dH1), ").")
        println(
            io, "- ``H_2`` is held best by **", best2.label, "** (", fmt(best2.dH2), ").")
        println(io, "- In the ``u`` chart the mass is conserved by every method, to ",
            fmt(maximum(r.dC0 for r in uchart)), " or better. Its gradient spans the ",
            "kernel of the first bracket, so any increment in that bracket's range ",
            "leaves it alone — explicit Euler included.")
        if !isempty(vchart)
            println(
                io, "- In the ``v`` chart it is **not** conserved, and drifts by up to ",
                fmt(maximum(r.dC0 for r in vchart)), ". That is not a defect of the ",
                raw"integrator: the Casimir of ``\mathbb{P}^1`` there is ",
                raw"``\int_\Omega v_h \, dx``, and the quantity that maps over is ",
                raw"``C_{0,d} = \int_\Omega v_h^2 \, dx``, which is not in the ",
                "kernel of anything.")
        end
        drifting = filter(r -> r.gH1 > 5 || r.gH2 > 5, rows)
        if isempty(drifting)
            println(io, "- Every error envelope is bounded: no growth ratio exceeds 5.")
        else
            println(io, "- Growing envelopes (ratio above 5, i.e. a secular drift rather ",
                "than a bounded oscillation): ",
                join(("**" * r.label * "**" for r in drifting), ", "), ".")
        end
        println(
            io, "- No method here holds both Hamiltonians. That is what the Ge-Marsden ",
            "theorem leads one to expect, rather than a gap in the list: a Poisson ",
            "integrator that also conserved the Hamiltonian exactly would reproduce ",
            "the exact flow up to a reparametrisation of time. The theorem's ",
            "non-degeneracy hypotheses are not checked for these systems, so it is ",
            "the reason for the trade-off and not a proof of it.")
        println(io)
        println(io, "## Figures")
        println(io)
        for g in ("flow1", "other")
            println(io, "**", g == "flow1" ? "flow 1" : "flow 2 and Miura", "**")
            println(io)
            for inv in ("H1", "H2", "C0")
                println(io, "- `kdv-$(case.key)-$(inv)-$(g).pdf`")
            end
            println(io, "- `kdv-$(case.key)-state-$(g).pdf`")
            println(io)
        end
    end
    note("  wrote $(basename(path))")
end

function run_case(case::Case)
    note("running $(case.key): $(case.title)" *
         (isempty(case.note) ? "" : "  ($(case.note))"))
    s = SplineSpace(case.n, case.p; L = case.L)
    sys = KdVSystem(s)

    # The `miura` case is posed in v, and the u system is then seeded from u₀ = M_h(v₀) so
    # that all eight runs start from the same field.
    #
    # Every OTHER case is posed in u, and used to have no Miura runs at all: at λ = 0 the
    # image of M_h is the half-space C₀ ≤ 0 and, sharply, the fields whose Hill operator is
    # positive definite, which excludes all of them. The spectral parameter removes that --
    # a preimage exists whenever λ < λ₀ -- so those cases now get Miura runs too, in a frame
    # translating at constant speed. See `miura_lambda`.
    local sysM, v0, u0
    if case.v0 !== nothing
        sysM = MiuraSystem(s)
        v0 = project(s, case.v0)
        u0 = miura_map(s, v0)
    else
        u0 = project(s, case.u0)
        λ = miura_lambda(s, u0)
        v0 = miura_invert(s, u0; λ = λ)
        if v0 === nothing
            note("no Miura preimage at λ = $(round(λ; digits = 4)); skipping the v-chart runs")
            sysM = nothing
        else
            sysM = MiuraSystem(s; λ = λ)
            note("Miura chart entered at λ = $(round(λ; digits = 4)) " *
                 "(λ₀ = $(round(hill_lambda0(s, u0); digits = 4)))")
        end
    end

    dt_rk4 = RK4_SAFETY * min(GeometricBrackets.stability_limit(sys.flow1, u0),
        GeometricBrackets.stability_limit(sys.flow2, u0))

    trajs, labels, finals = Trajectory[], String[], Vector{Float64}[]
    rows = RunSummary[]

    for (label, flow, meth, explicit) in runs_u(sys)
        # The stability limit, not `min(dt_rk4, case.dt)`: on the two wide benchmarks it is
        # LARGER than the step size the implicit methods want, and clamping it there would
        # hide the very point the comparison makes -- that on those two examples an explicit
        # method is not paying a step-count penalty at all. This matches
        # `plot_kdv_energies.py`, which uses `dt_rk4` outright.
        dt = explicit ? dt_rk4 : case.dt
        traj, NT = run_one(sys, flow, meth, u0, dt, case.T)
        push!(trajs, traj)
        push!(labels, label)
        push!(finals, traj.final)
        push!(rows, summarise(label, traj, dt, NT))
        report(label, traj)
    end

    if sysM !== nothing
        for (label, flow, meth, _) in runs_v(sysM)
            traj, NT = run_one(sysM, flow, meth, v0, case.dt, case.T)
            push!(trajs, traj)
            push!(labels, label)
            # the state of a Miura run is v̂; map it before plotting alongside the others
            push!(finals, miura_map(sysM, traj.final))
            push!(rows, summarise(label, traj, case.dt, NT))
            report(label, traj)
        end
    end

    mkpath(FIGDIR)

    # One invariant per figure, one family of vector fields per figure. Six energy figures
    # rather than one panel of everything: the errors of the three invariants sit orders of
    # magnitude apart, and the comparison that matters is within a family, not across. The
    # state figure is split the same way and named the same way.
    for g in (:flow1, :other)
        keep = [i for i in eachindex(labels) if group_of(labels[i]) == g]
        isempty(keep) && continue
        gname = string(g)

        for inv in (:H1, :H2, :C0)
            save(joinpath(FIGDIR, "kdv-$(case.key)-$(inv)-$(gname).pdf"),
                energyplot(trajs[keep], labels[keep];
                    name = inv,
                    title = "$(case.title) — $(GROUP_TITLE[g])"))
        end

        save(joinpath(FIGDIR, "kdv-$(case.key)-state-$(gname).pdf"),
            stateplot(sys, vcat([u0], finals[keep]), vcat(["initial"], labels[keep]);
                xleft = case.xleft,
                title = "$(case.title) — $(GROUP_TITLE[g]): initial and final states"))
    end

    write_summary(case, rows, s)
    note("  wrote 6 energy figures and 2 state figures for $(case.key)")
end

let cases = filter(a -> !startswith(a, "--"), ARGS)
    for k in (isempty(cases) ? [c.key for c in CASES] : cases)
        run_case(find_case(k))
    end
end
