module GeometricBracketsMakieExt

using CairoMakie
using Printf
using GeometricBrackets
using GeometricBrackets: ERROR_FLOOR, FLOW_STYLES, INTEGRATOR_COLORS

"""
    _series_style(label)

Resolve a run label of the form `"<integrator>, flow <n>"` into a colour and a line style.

Colour carries the integrator and line style the vector field, so that two runs of the same
method on different flows are told apart by shape rather than by hue, and no series in the
figure is identified by colour alone.
"""
function _series_style(label::AbstractString)
    parts = split(label, ',')
    name = strip(parts[1])
    color = get(INTEGRATOR_COLORS, name, "#666666")
    flow = 1
    for p in parts[2:end]
        q = strip(p)
        occursin("Miura", q) && (flow = :M; continue)
        m = match(r"flow\s*(\d+)", q)
        m === nothing || (flow = parse(Int, m.captures[1]))
    end
    (color = color, linestyle = get(FLOW_STYLES, flow, :solid))
end

_clamped(v) = max.(v, ERROR_FLOOR)

function GeometricBrackets.energyplot(trajs, labels;
        name::Symbol = :H1,
        title = "",
        ylabel = nothing,
        size = (620, 380),
        resolution = nothing)
    isempty(trajs) && throw(ArgumentError("no trajectories to plot"))
    length(trajs) == length(labels) || throw(DimensionMismatch(
        "got $(length(trajs)) trajectories but $(length(labels)) labels"))

    # A reference value that vanishes -- the mass of the cosine initial condition is zero to
    # round-off -- makes a relative error meaningless, so the whole panel switches to the
    # absolute one rather than reporting a 100-fold "drift" of a conserved quantity.
    refs = [abs(getproperty(t.reference, name)) for t in trajs]
    relative = all(r -> r > sqrt(eps(Float64)), refs)

    fig = Figure(size = resolution === nothing ? size : resolution)
    ax = Axis(fig[1, 1];
        xlabel = "t",
        ylabel = ylabel === nothing ?
                 (relative ? "relative error" : "absolute error") : ylabel,
        title = isempty(title) ? string(name) : "$(title):  $(name)",
        yscale = log10)

    for (traj, label) in zip(trajs, labels)
        st = _series_style(label)
        d = deviation(traj, name)
        y = relative ? d ./ abs(getproperty(traj.reference, name)) : d
        lines!(ax, traj.t, _clamped(y);
            color = st.color, linestyle = st.linestyle, label = label)
    end

    Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
    fig
end

function GeometricBrackets.stateplot(system, states, labels;
        npoints = 601,
        title = "",
        xleft = 0.0,
        size = (900, 400),
        resolution = nothing)
    length(states) == length(labels) || throw(DimensionMismatch(
        "got $(length(states)) states but $(length(labels)) labels"))

    s = system.space
    L = domainlength(s)
    xs = collect(range(0, L, length = npoints))

    fig = Figure(size = resolution === nothing ? size : resolution)
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u")

    # Nested strokes of decreasing width, so that curves lying on top of one another all
    # stay visible instead of the last one drawn hiding the rest. The widths are spread
    # over the whole range for however many runs there are -- zipping against a fixed
    # tuple would silently truncate, and the extra runs of the Miura case would vanish
    # from the figure without a word.
    widths = length(states) == 1 ? [2.0] :
             collect(range(2.8, 0.6, length = length(states)))
    for (i, (û, label)) in enumerate(zip(states, labels))
        st = _series_style(label)
        lines!(ax, xs .+ xleft, evaluate(s, û, xs);
            color = st.color, linestyle = st.linestyle,
            linewidth = widths[i], label = label)
    end

    # The legend goes BESIDE the axes, not inside them: the final states of these runs lie
    # almost on top of one another and fill the frame, so an inset legend covers the very
    # part of the curve the figure is about.
    Legend(fig[1, 2], ax; framevisible = false)
    isempty(title) || Label(fig[0, 1:2], title; fontsize = 16)
    fig
end

"""
    energyplot(trajs, labels, names; kwargs...)

A row of panels, one per invariant in `names`, sharing a legend.

The single-panel method is still the right default -- errors in different invariants sit
orders of magnitude apart and a shared axis flatters whichever is worst. This method exists
for the manuscript's own figure layout, which puts H1, H2 and C0 side by side, and each panel
keeps its own y axis.
"""
function GeometricBrackets.energyplot(trajs, labels, names;
        title = "",
        size = (300 * length(names) + 160, 380),
        resolution = nothing)
    isempty(names) && throw(ArgumentError("no invariants to plot"))
    fig = Figure(size = resolution === nothing ? size : resolution)
    local ax
    for (col, nm) in enumerate(names)
        refs = [abs(getproperty(t.reference, nm)) for t in trajs]
        relative = all(r -> r > sqrt(eps(Float64)), refs)
        ax = Axis(fig[1, col];
            xlabel = "t",
            ylabel = col == 1 ? (relative ? "relative error" : "absolute error") : "",
            title = string(nm),
            yscale = log10)
        col == 1 || hideydecorations!(ax; grid = false)
        for (traj, label) in zip(trajs, labels)
            st = _series_style(label)
            d = deviation(traj, nm)
            y = relative ? d ./ abs(getproperty(traj.reference, nm)) : d
            lines!(ax, traj.t, _clamped(y); color = st.color, linestyle = st.linestyle,
                label = label)
        end
    end
    isempty(title) || Label(fig[0, :], title; halign = :left, font = :bold)
    Legend(fig[2, :], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
    fig
end

"A neutral grey for reference guides, distinct from every Okabe-Ito series colour."
const GUIDE_GREY = "#8c8c8c"

"Least-squares slope of log y against log x, for the fitted orders in the legends."
function _order(xs, ys)
    keep = [i for i in eachindex(ys) if ys[i] > 0 && isfinite(ys[i])]
    length(keep) < 2 && return NaN
    lx, ly = log.(xs[keep]), log.(ys[keep])
    x̄, ȳ = sum(lx) / length(lx), sum(ly) / length(ly)
    sum((lx .- x̄) .* (ly .- ȳ)) / sum((lx .- x̄) .^ 2)
end

function GeometricBrackets.sweepplot(dts, series;
        title = "",
        xlabel = "Δt",
        size = (620, 520),
        resolution = nothing)
    isempty(series) && throw(ArgumentError("no series to plot"))
    all(length(last(sv)) == length(dts) for sv in series) || throw(DimensionMismatch(
        "every series must have one value per step size"))

    fig = Figure(size = resolution === nothing ? size : resolution)
    ax1 = Axis(fig[1, 1]; ylabel = "departure", xscale = log10, yscale = log10,
        title = title)
    ax2 = Axis(fig[2, 1]; ylabel = "excess over the plateau", xlabel = xlabel,
        xscale = log10, yscale = log10)
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false)

    ref = nothing
    for (label, values) in series
        st = _series_style(label)
        v = collect(float.(values))
        lines!(ax1, dts, _clamped(v); color = st.color,
            linestyle = st.linestyle, label = label)
        scatter!(ax1, dts, _clamped(v); color = st.color, markersize = 5)
        # each run's own plateau is its value at the smallest step size
        excess = _clamped(v[1:(end - 1)] .- v[end])
        lines!(ax2, dts[1:(end - 1)], excess; color = st.color, linestyle = st.linestyle,
            label = label)
        scatter!(ax2, dts[1:(end - 1)], excess; color = st.color, markersize = 5)
        if ref === nothing && v[1] - v[end] > 0
            ref = v[1] - v[end]
        end
    end

    if ref !== nothing
        guide = [1.35 * ref * (dt / dts[1])^2 for dt in dts[1:(end - 1)]]
        lines!(ax2, dts[1:(end - 1)], _clamped(guide); color = GUIDE_GREY,
            linestyle = (:dot, :dense), linewidth = 1.0, label = "O(Δt²)")
    end

    Legend(fig[3, 1], ax2; orientation = :horizontal, nbanks = 2, framevisible = false)
    fig
end

function GeometricBrackets.convergenceplot(xs, series;
        title = "",
        xlabel = "N",
        ylabel = "error",
        guide = nothing,
        size = (620, 400),
        resolution = nothing)
    isempty(series) && throw(ArgumentError("no series to plot"))
    all(length(last(sv)) == length(xs) for sv in series) || throw(DimensionMismatch(
        "every series must have one value per resolution"))

    fig = Figure(size = resolution === nothing ? size : resolution)
    # the resolutions rarely span a decade, so label the actual values rather than
    # letting Makie print 10^1.1, 10^1.2, ...
    ax = Axis(fig[1, 1]; xlabel, ylabel, title, xscale = log10, yscale = log10,
        xticks = (collect(float.(xs)), string.(xs)))

    for (label, values) in series
        st = _series_style(label)
        v = _clamped(collect(float.(values)))
        q = _order(collect(float.(xs)), v)
        # A flat series is the interesting case in half of these figures, so report the
        # fitted order rather than assuming there is a rate to quote.
        # `+ 0.0` normalises the -0.0 that a flat series would otherwise print
        lab = isnan(q) ? label :
              @sprintf("%s  (order %.1f)", label, round(-q; digits = 1) + 0.0)
        lines!(ax, xs, v; color = st.color, linestyle = st.linestyle, label = lab)
        scatter!(ax, xs, v; color = st.color, markersize = 5)
    end

    if guide !== nothing
        y0 = maximum(maximum(float.(last(sv))) for sv in series)
        g = [y0 * (float(x) / float(xs[1]))^(-guide) for x in xs]
        lines!(ax, xs, _clamped(g); color = GUIDE_GREY, linestyle = (:dot, :dense),
            linewidth = 1.0, label = @sprintf("O(h^%g)", guide))
    end

    axislegend(ax; position = :lb, framevisible = false)
    fig
end

end
