using GeometricBrackets
using Documenter

DocMeta.setdocmeta!(GeometricBrackets, :DocTestSetup,
    :(using GeometricBrackets); recursive = true)

makedocs(;
    modules = [GeometricBrackets],
    authors = "Michael Kraus",
    repo = Remotes.GitHub("JuliaGNI", "GeometricBrackets.jl"),
    sitename = "GeometricBrackets.jl",
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaGNI.github.io/GeometricBrackets.jl",
        # set explicitly: without a configured remote, Documenter cannot read the branch
        # from `git remote` and would default to "master"
        edit_link = "main",
        # the library page collects every docstring in the package and is legitimately
        # large; the default warning threshold is aimed at pages that are large by accident
        size_threshold_warn = 400 * 1024,
        size_threshold = 800 * 1024,
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Discretisation" => "discretisation.md",
        "Equations" => ["Korteweg-de Vries" => "kdv.md",
            "Camassa-Holm" => "camassaholm.md",
            "Burgers" => "burgers.md"],
        "Discrete Lie-Poisson brackets" => ["Structure constants" => "liepoisson.md",
            "Dirac reduction" => "dirac.md"],
        "Four-brackets" => "fourbrackets.md",
        "Integrators" => "integrators.md",
        "Diagnostics" => "diagnostics.md",
        "Backward error analysis" => "bea.md",
        "No-go results" => ["Aliasing and the zero-mode theorem" => "aliasing.md",
            "Antisymmetric three-brackets" => "nambu.md"],
        "Verification" => ["What was checked" => "verification.md",
            "The scripts" => "scripts.md"],
        "Library" => "library.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGNI/GeometricBrackets.jl",
    devurl = "latest",
    devbranch = "main"
)
