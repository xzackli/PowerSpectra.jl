using PowerSpectra
using Documenter

makedocs(;
    modules=[PowerSpectra],
    authors="Zack Li",
    repo="https://github.com/xzackli/PowerSpectra.jl/blob/{commit}{path}#L{line}",
    sitename="PowerSpectra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xzackli.github.io/PowerSpectra.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Spectra" => "spectra.md",
        "Covariance" => "covariance.md",
        "Beams" => "beams.md",
        "Utilities" => "util.md",
        "Index" => "module_index.md",
    ],
)

deploydocs(;
    repo="github.com/xzackli/PowerSpectra.jl",
)
