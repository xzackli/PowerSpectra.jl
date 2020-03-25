using Documenter, PSPlanck

makedocs(;
    modules=[PSPlanck],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/xzackli/PSPlanck.jl/blob/{commit}{path}#L{line}",
    sitename="PSPlanck.jl",
    authors="xzackli",
    assets=String[],
)

deploydocs(;
    repo="github.com/xzackli/PSPlanck.jl",
)
