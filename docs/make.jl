using Documenter, PlanckCov

makedocs(;
    modules=[PlanckCov],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/xzackli/PlanckCov.jl/blob/{commit}{path}#L{line}",
    sitename="PlanckCov.jl",
    authors="xzackli",
    assets=String[],
)

deploydocs(;
    repo="github.com/xzackli/PlanckCov.jl",
)
