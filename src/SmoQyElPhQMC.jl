module SmoQyElPhQMC

using LatticeUtilities
using LinearAlgebra
using FFTW
using Random
using Statistics
using ShiftedArrays
using CircularArrays
using StaticArrays


# Reshapes with zero allocations, returns an instance of Base.ReshapedArray.
# Discussion found at: https://github.com/JuliaLang/julia/issues/36313
reshaped(a::AbstractArray, dims...) = reshaped(a, dims)
reshaped(a::Base.ReshapedArray, dims::NTuple{N,Int}) where {N} = reshaped(a.parent, dims)
reshaped(a::AbstractArray, dims::NTuple{N,Int}) where {N} = Base.ReshapedArray(a, dims, ())

# function reshaped(a::AbstractArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
#     return (size(a) == dims) ? a : invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
# end

# function reshaped(a::AbstractArray{T,M}, dims...) where {T,M}
#     return reshaped(a, dims)
# end

# import function for calculating checkerboard decomosition
import Checkerboard: checkerboard_decomposition!

# for representing checkerboard matrix
using Checkerboard

# import for propagator types
using JDQMCFramework

# import for some measurement routines
import JDQMCMeasurements: fourier_transform!

# functions to overload
import Base: size, eltype
import LinearAlgebra: mul!, lmul!, ldiv!

# import SmoQyDQMC
using SmoQyDQMC
import SmoQyDQMC: HolsteinParameters, SSHParameters, DispersionParameters

# using SmoQyKPMCore for preconditioner
using SmoQyKPMCore

include("IterativeSolvers/ConjugateGradient.jl")

# implement function for perform in-place checkerboard matrix-vector multiply
include("checkerboard_matrix_multiply.jl")

include("FermionDetMatrix.jl")
export FermionDetMatrix, SymFermionDetMatrix, AsymFermionDetMatrix

include("holstein_shift_matrix.jl")

include("fermion_det_matrix_dervative.jl")

include("FourierTransformer.jl")

include("KPMPreconditioner.jl")
export KPMPreconditioner, SymKPMPreconditioner, AsymKPMPreconditioner

import SmoQyDQMC: hmc_update!
include("EFAPFFHMCUpdater.jl")
export EFAPFFHMCUpdater

import SmoQyDQMC: reflection_update!
include("reflection_update.jl")

import SmoQyDQMC: swap_update!
include("swap_update.jl")

include("Measurements/GreensEstimator.jl")
export GreensEstimator

include("Measurements/scalar_measurements.jl")

import SmoQyDQMC: measure_onsite_energy, measure_hopping_energy, measure_bare_hopping_energy
include("Measurements/tight_binding_measurements.jl")

import SmoQyDQMC: measure_holstein_energy, measure_ssh_energy
include("Measurements/electron_phonon_measurements.jl")

import SmoQyDQMC: make_measurements!, CorrelationContainer, CompositeCorrelationContainer
include("Measurements/Correlations/density.jl")
include("Measurements/Correlations/pair.jl")
include("Measurements/Correlations/spin.jl")
include("Measurements/Correlations/bond.jl")
include("Measurements/Correlations/current.jl")
include("Measurements/make_measurements.jl")

end
