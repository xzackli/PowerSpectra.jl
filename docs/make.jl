using AngularPowerSpectrum
using Documenter

makedocs(;
    modules=[AngularPowerSpectrum],
    authors="xzackli <xzackli@gmail.com> and contributors",
    repo="https://github.com/xzackli/AngularPowerSpectrum.jl/blob/{commit}{path}#L{line}",
    sitename="AngularPowerSpectrum.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xzackli.github.io/AngularPowerSpectrum.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/xzackli/AngularPowerSpectrum.jl",
)
