module SmoQyElPhQMC

using LatticeUtilities
using LinearAlgebra
using FFTW
using Random
using Statistics
using ShiftedArrays
using CircularArrays


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

# functions to overload
import Base: size, eltype
import LinearAlgebra: mul!, lmul!, ldiv!

# import SmoQyDQMC
using SmoQyDQMC

# import function to extend from SmoQyDQMC
import SmoQyDQMC: hmc_update!

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

include("EFAPFFHMCUpdater.jl")
export EFAPFFHMCUpdater

include("Measurements/GreensEstimator.jl")
export GreensEstimator

include("Measurements/scalar_measurements.jl")

import SmoQyDQMC: measure_onsite_energy, measure_hopping_energy, measure_bare_hopping_energy
include("Measurements/tight_binding_measurements.jl")

import SmoQyDQMC: make_measurements!
include("Measurements/make_measurements.jl")

end
