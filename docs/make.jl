using SmoQyElPhQMC
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Literate
using SmoQyDQMC

# Remove existing Documenter `build` directory
build_path = joinpath(@__DIR__, "build")
isdir(build_path) && rm(build_path; recursive=true)

# Create `build/assets` directories
scripts_path = joinpath(build_path, "assets", "scripts")
example_scripts_path = joinpath(scripts_path, "examples")
tutorial_scripts_path = joinpath(scripts_path, "tutorials")
mkpath.([example_scripts_path, tutorial_scripts_path])

# generates script and notebook versions of tutorials based on literate example
function build_examples(example_sources, destdir)
    assetsdir = joinpath(fill("..", length(splitpath(destdir)))..., "assets")

    script_type = destdir
    destpath = joinpath(@__DIR__, "src", destdir)
    isdir(destpath) && rm(destpath; recursive=true)

    # Transform each Literate source file to Markdown for subsequent processing by
    # Documenter.
    for source in example_sources
        # Extract "example" from "path/example.jl"
        name = splitext(basename(source))[1]
        
        # Preprocess each example by adding a notebook download link at the top. The
        # relative path is hardcoded according to the layout of `gh-pages` branch,
        # which is set up by `Documenter.deploydocs`.
        function preprocess(str)
            """
            # Download this example as a [Julia script]($assetsdir/scripts/$script_type/$name.jl).

            """ * str
        end
        # Write to `src/$destpath/$name.md`
        Literate.markdown(
            source, destpath;
            preprocess,
            credit=false,
            codefence = "````julia" => "````"
        )
    end

    # Create Julia script for each Literate example.
    # These will be stored in the `assets/` directory of the hosted docs.
    for source in example_sources

        # Build julia scripts
        Literate.script(
            source, joinpath(scripts_path, script_type);
            credit=false
        )
    end

    # Return paths `$destpath/$name.md` for each new Markdown file (relative to
    # `src/`)
    return map(example_sources) do source
        name = splitext(basename(source))[1]
        joinpath(destdir, "$name.md")
    end
end

# initialize bibliography
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style=:numeric
)

DocMeta.setdocmeta!(SmoQyElPhQMC, :DocTestSetup, :(using SmoQyElPhQMC); recursive=true)

# build tutorials
tutorials = ["holstein_honeycomb"]
tutorials_sources = [joinpath(pkgdir(SmoQyElPhQMC, "tutorials"), tutorial*".jl") for tutorial in tutorials]
tutorial_mds = build_examples(tutorials_sources, "tutorials")

# build examples
examples = []
example_sources = [joinpath(pkgdir(SmoQyElPhQMC, "examples"), example*".jl") for example in examples]
example_mds = build_examples(example_sources, "examples")

# link to external package APIs
links = InterLinks(
    "SmoQyDQMC" => "https://smoqysuite.github.io/SmoQyDQMC.jl/stable/",
    "LatticeUtilities" => "https://smoqysuite.github.io/LatticeUtilities.jl/stable/"
)

makedocs(;
    clean = false,
    modules=[SmoQyElPhQMC],
    plugins=[bib,links],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>, Steven Johnston <sjohn145@utk.edu>",
    sitename="SmoQyElPhQMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/SmoQyElPhQMC.jl/stable",
        edit_link="main",
        assets=String[],
        size_threshold_warn = 1000*1024, # 200KB -- library.html gets quite large
        size_threshold      = 2000*2024, # 300KB
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Tutorials" => tutorial_mds,
        # "Examples" => examples_mds,
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyElPhQMC.jl",
    devbranch="main",
)
