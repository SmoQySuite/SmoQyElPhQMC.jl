# type for applying basis change to fermion determinant matrix to frequency space
struct FourierTransformer{T<:AbstractFloat,Tfft<:AbstractFFTs.Plan,Tifft<:AbstractFFTs.Plan}

    Lτ::Int
    N::Int
    θ::Vector{Complex{T}}
    fft_plan!::Tfft
    ifft_plan!::Tifft
end

# initialize fourier transformer
function FourierTransformer(T::DataType, Lτ::Int, N::Int)

    @assert T <: AbstractFloat
    θ = [exp(-im*π*(l-1)/Lτ) for l in 1:Lτ]
    u = zeros(Complex{T}, Lτ, N)
    fft_plan!  = plan_fft!(u, (1,), flags=FFTW.PATIENT)
    ifft_plan! = plan_ifft!(u, (1,), flags=FFTW.PATIENT)

    return FourierTransformer(Lτ, N, θ, fft_plan!, ifft_plan!)
end

FourierTransformer(v::Matrix{T}) where {T<:Number} = FourierTransformer(real(T), size(v)...)

# evalute u = U⋅v, τ → ω
function mul!(
    u::AbstractVecOrMat{Complex{T}},
    U::FourierTransformer{T},
    v::AbstractVecOrMat{E}
) where {T<:AbstractFloat, E<:Number}
    
    copyto!(u, v)
    lmul!(U, u)

    return nothing
end

# evaluate v = U⋅v, τ → ω
function lmul!(
    U::FourierTransformer{T},
    v::AbstractVecOrMat{Complex{T}}
) where {T<:AbstractFloat}
    
    (; Lτ, N, θ, fft_plan!) = U
    u = (ndims(v)==2) ? v : reshaped(v, Lτ, N)
    @. u = θ/sqrt(Lτ) * u
    mul!(u, fft_plan!, u)

    return nothing
end

# evaluate v = U⁻¹⋅v, ω → τ
function ldiv!(
    U::FourierTransformer{T},
    v::AbstractVecOrMat{Complex{T}}
) where {T<:AbstractFloat}
    
    (; Lτ, N, θ, ifft_plan!) = U
    u = (ndims(v)==2) ? v : reshaped(v, Lτ, N)
    mul!(u, ifft_plan!, u)
    @. u = inv(θ)*sqrt(Lτ) * u

    return nothing
end

# evaluate u = U⁻¹⋅v, ω → τ
function ldiv!(
    u::AbstractVecOrMat{Complex{T}},
    U::FourierTransformer{T},
    v::AbstractVecOrMat{E}
) where {T<:AbstractFloat, E<:Number}
    
    copyto!(u, v)
    ldiv!(U, u)

    return nothing
end