module SmoQyElPhQMC

using LatticeUtilities
using LinearAlgebra
using FFTW
using Random
using Statistics
using ShiftedArrays
using CircularArrays
using StaticArrays
using PkgVersion

# get and set package version number as global constant
const SMOQYELPHQMC_VERSION = PkgVersion.@Version 0

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

# import function for calculating checkerboard decomposition
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

# define types for representing fermion determinant matrix
include("FermionDetMatrix.jl")
export FermionDetMatrix, SymFermionDetMatrix, AsymFermionDetMatrix

# Define the holstein shift matrix that transform Λ that transforms the
# fermion determinant matrix to M → M⋅Λ that arrises as a result of parameterizing
# the holstein interaction g⋅X⋅(n-1).
# See Phys. Rev. E 105, 065302 for more information.
include("holstein_shift_matrix.jl")

# methods for calculating ⟨u|∂M/∂x|v⟩ where ∂M/∂x is the derivative of the fermion determinant matrix
include("fermion_det_matrix_dervative.jl")

# type to apply unitary transformation to fermion determinant matrix to go from
# imaginary time to frequency basis
include("FourierTransformer.jl")

# KPM based preconditioner to accelerate conjugate gradient solves
include("KPMPreconditioner.jl")
export KPMPreconditioner, SymKPMPreconditioner, AsymKPMPreconditioner

# type to manage sampling and storing the pseudo-fermion field Φ
# as well as calculating the fermionic action
include("PFFCalculator.jl")
export PFFCalculator

# For performing HMC updates to phonon fields that use exact fourier acceleration (EFA)
# to improve sampling
import SmoQyDQMC: hmc_update!
include("EFAPFFHMCUpdater.jl")
export EFAPFFHMCUpdater

# perform linear scaling reflection update to phonon fields
import SmoQyDQMC: reflection_update!
include("reflection_update.jl")

# perform linear scaling swap update to phonon fields
import SmoQyDQMC: swap_update!
include("swap_update.jl")

# perform radial update to phonon fields
import SmoQyDQMC: radial_update!
include("radial_update.jl")

# type of estimate Green's function and Green's function Wick's contractions
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

import MuTuner
import MuTuner: MuTunerLogger
import SmoQyDQMC: update_chemical_potential! 
include("update_chemical_potential.jl")

end
