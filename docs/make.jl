using PowerSpectra
using Documenter

ENV["PLOTS_DEFAULT_BACKEND"] = "GR"
ENV["GKSwstype"]="nul"
const PLOTS_DEFAULTS = Dict(:theme=>:default, :fontfamily => "Computer Modern",
    :linewidth=>1.5,
    :titlefontsize=>(16+8), :guidefontsize=>(11+5), 
    :tickfontsize=>(8+4), :legendfontsize=>(8+4),
    # :left_margin=>5mm, :right_margin=>5mm
    )
using Plots

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
        "Quickstart" => "quickstart.md",
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
