# API

## Fermion Determinant Matrix

- [`FermionDetMatrix`](@ref)
- [`SymFermionDetMatrix`](@ref)
- [`AsymFermionDetMatrix`](@ref)

```@docs
FermionDetMatrix
SymFermionDetMatrix
SymFermionDetMatrix(::FermionPathIntegral{T, E})  where {T<:Number, E<:AbstractFloat}
AsymFermionDetMatrix
AsymFermionDetMatrix(::FermionPathIntegral{T, E})  where {T<:Number, E<:AbstractFloat}
```

## Preconditioners

- [`KPMPreconditioner`](@ref)
- [`SymKPMPreconditioner`](@ref)
- [`AsymKPMPreconditioner`](@ref)

```@docs
KPMPreconditioner
KPMPreconditioner(::FermionDetMatrix{T}) where {T<:Number, E<:AbstractFloat}
SymKPMPreconditioner
AsymKPMPreconditioner
```

## Monte Carlo Update Methods

- [`PFFCalculator`](@ref)
- [`EFAPFFHMCUpdater`](@ref)
- [`hmc_update!`](@ref)
- [`reflection_update!`](@ref)
- [`swap_update!`](@ref)
- [`radial_update!`](@ref)

```@docs
PFFCalculator
PFFCalculator(::ElectronPhononParameters{T, E}, ::FermionDetMatrix{T, E}) where {T<:Number, E<:AbstractFloat}
EFAPFFHMCUpdater
EFAPFFHMCUpdater(;)
hmc_update!
reflection_update!
swap_update!
radial_update!
```

## Measurement Methods

- [`GreensEstimator`](@ref)
- [`make_measurements!`](@ref)

```@docs
GreensEstimator
GreensEstimator(::FermionDetMatrix{T,E}, ::ModelGeometry{D,E}) where {D, T<:Number, E<:AbstractFloat}
make_measurements!
```

## Chemical Potential Tuning

- [`update_chemical_potential!`](@ref)

```@docs
update_chemical_potential!
```