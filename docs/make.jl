using AngularPowerSpectra
using Documenter

makedocs(;
    modules=[AngularPowerSpectra],
    authors="xzackli <xzackli@gmail.com> and contributors",
    repo="https://github.com/xzackli/AngularPowerSpectra.jl/blob/{commit}{path}#L{line}",
    sitename="AngularPowerSpectra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xzackli.github.io/AngularPowerSpectra.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/xzackli/AngularPowerSpectra.jl",
)
