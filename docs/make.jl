using SmoQyElPhQMC
using Documenter

DocMeta.setdocmeta!(SmoQyElPhQMC, :DocTestSetup, :(using SmoQyElPhQMC); recursive=true)

makedocs(;
    modules=[SmoQyElPhQMC],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>, Steven Johnston <sjohn145@utk.edu>",
    sitename="SmoQyElPhQMC.jl",
    format=Documenter.HTML(;
        canonical="https://SmoQySuite.github.io/SmoQyElPhQMC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyElPhQMC.jl",
    devbranch="main",
)
