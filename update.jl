using Pkg

Pkg.activate(@__DIR__)
@info "updating $(Pkg.project().name)..."
Pkg.update()

Pkg.activate(joinpath(@__DIR__, "ExperimentTools"))
@info "updating $(Pkg.project().name)..."
Pkg.update()

Pkg.activate(joinpath(@__DIR__, "experiments"))
@info "updating experiments..."
Pkg.update()
