@doc raw"""
    AbstractFermionDetMatrix{T<:Number, E<:AbstractFloat}

A abstract type to represent fermion determinant matrix
```
M = \left(\begin{array}{ccccc}
    I &  &  &  & B_{0}\\
    -B_{1} & I\\
    & -B_{2} & \ddots\\
    &  & \ddots & \ddots\\
    &  &  & -B_{L_{\tau}-1} & I
\end{array}\right),
```
where ``B_l`` are propagator matrices for imaginary-time slice ``\tau = \Delta\tau \cdot l`` given
an inverse temperature ``\beta = \Delta\tau \cdot L_\tau``. A Fermion determinant matrix ``M``
will be ``N L_\tau \times N L_\tau``, where each propagator matrix ``B_l`` is ``N \times N``,
where ``N`` is the number of orbitals in the lattice.
"""
abstract type AbstractFermionDetMatrix{T<:Number, E<:AbstractFloat} end

@doc raw"""
    SymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: AbstractFermionDetMatrix{T,E}

A type to represent fermion determinant matrix
```
M = \left(\begin{array}{ccccc}
    I &  &  &  & B_{0}\\
    -B_{1} & I\\
    & -B_{2} & \ddots\\
    &  & \ddots & \ddots\\
    &  &  & -B_{L_{\tau}-1} & I
\end{array}\right),
```
where
```
B_l = \left[ e^{-\Delta\tau K_l/2} \right]^\dagger e^{-\Delta\tau V_l} e^{-\Delta\tau K_l/2}
```
are Hermitian (symmetric if real) propagator matrices for imaginary-time slice ``\tau = \Delta\tau \cdot l`` given
an inverse temperature ``\beta = \Delta\tau \cdot L_\tau``. A Fermion determinant matrix ``M``
will be ``N L_\tau \times N L_\tau``, where each propagator matrix ``B_l`` is ``N \times N``,
where ``N`` is the number of orbitals in the lattice. Here the matrix ``e^{-\Delta\tau K_l/2}`` is not Hermitian
as it is approximated using the checkerboard approximation.
"""
struct SymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: AbstractFermionDetMatrix{T,E}

    expnΔτV::Matrix{E}
    coshΔτt::Matrix{T}
    sinhΔτt::Matrix{T}
    checkerboard_neighbor_table::Matrix{Int}
    checkerboard_perm::Vector{Int}
    checkerboard_colors::Vector{UnitRange{Int}}
    rtmp1::Matrix{E}
    rtmp2::Matrix{E}
    ztmp1::Matrix{Complex{E}}
    ztmp2::Matrix{Complex{E}}
end

function SymFermionDetMatrix(fpi::FermionPathIntegral{T}) where {T<:Number}

    (; neighbor_table, t, V, N, β, Δτ, Lτ) = fpi

    # get number of hoppings
    Nh = size(t, 1)

    # allocate arrays
    expnΔτV = zeros(real(T), Lτ, N)
    coshΔτt = zeros(T, Lτ, Nh)
    sinhΔτt = zeros(T, Lτ, Nh)

    # allocate temorary storage vector
    rtmp1 = zeros(real(T), Lτ, N)
    rtmp2 = zeros(real(T), Lτ, N)
    ztmp1 = zeros(Complex{real(T)}, Lτ, N)
    ztmp2 = zeros(Complex{real(T)}, Lτ, N)

    # construct checkerboard decomposition
    if iszero(Nh)
        checkerboard_neighbor_table = Matrix{Int}(undef,(2,0))
        checkerboard_perm = Int[]
        checkerboard_color_intervals = UnitRange{Int}[]
    else
        checkerboard_neighbor_table = deepcopy(neighbor_table)
        checkerboard_perm, checkerboard_colors = checkerboard_decomposition!(checkerboard_neighbor_table)
        checkerboard_color_intervals = [checkerboard_colors[1,i]:checkerboard_colors[2,i] for i in axes(checkerboard_colors,2)]
    end

    # allocate FermionDetMatrixMultiplier
    sym_fdm = SymFermionDetMatrix{T,real(T)}(
        expnΔτV, coshΔτt, sinhΔτt,
        checkerboard_neighbor_table, checkerboard_perm, checkerboard_color_intervals,
        rtmp1, rtmp2, ztmp1, ztmp2
    )

    # upate FermionDetMatrixMultiplier
    update!(sym_fdm, fpi)

    return sym_fdm
end


@doc raw"""
    AsymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: AbstractFermionDetMatrix{T, E}

A type to represent fermion determinant matrix
```
M = \left(\begin{array}{ccccc}
    I &  &  &  & B_{0}\\
    -B_{1} & I\\
    & -B_{2} & \ddots\\
    &  & \ddots & \ddots\\
    &  &  & -B_{L_{\tau}-1} & I
\end{array}\right),
```
where
```
B_l = e^{-\Delta\tau V_l} e^{-\Delta\tau K_l}
```
are Hermitian (symmetric if real) propagator matrices for imaginary-time slice ``\tau = \Delta\tau \cdot l`` given
an inverse temperature ``\beta = \Delta\tau \cdot L_\tau``. A Fermion determinant matrix ``M``
will be ``N L_\tau \times N L_\tau``, where each propagator matrix ``B_l`` is ``N \times N``,
where ``N`` is the number of orbitals in the lattice. Note that``e^{-\Delta\tau K_l}`` is
represented using the checkerboard approximation.
"""
struct AsymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: AbstractFermionDetMatrix{T,E}

    expnΔτV::Matrix{E}
    coshΔτt::Matrix{T}
    sinhΔτt::Matrix{T}
    checkerboard_neighbor_table::Matrix{Int}
    checkerboard_perm::Vector{Int}
    checkerboard_colors::Vector{UnitRange{Int}}
    rtmp1::Matrix{E}
    rtmp2::Matrix{E}
    ztmp1::Matrix{Complex{E}}
    ztmp2::Matrix{Complex{E}}
end

function AsymFermionDetMatrix(fpi::FermionPathIntegral{T}) where {T<:Number}

    (; neighbor_table, t, N, Lτ) = fpi

    # get number of hoppings
    Nh = size(t, 1)

    # allocate arrays
    expnΔτV = zeros(real(T), Lτ, N)
    coshΔτt = zeros(T, Lτ, Nh)
    sinhΔτt = zeros(T, Lτ, Nh)

    # allocate temorary storage vector
    rtmp1 = zeros(real(T), Lτ, N)
    rtmp2 = zeros(real(T), Lτ, N)
    ztmp1 = zeros(Complex{real(T)}, Lτ, N)
    ztmp2 = zeros(Complex{real(T)}, Lτ, N)

    # construct checkerboard decomposition
    if iszero(Nh)
        checkerboard_neighbor_table = Matrix{Int}(undef,(2,0))
        checkerboard_perm = Int[]
        checkerboard_color_intervals = UnitRange{Int}[]
    else
        checkerboard_neighbor_table = deepcopy(neighbor_table)
        checkerboard_perm, checkerboard_colors = checkerboard_decomposition!(checkerboard_neighbor_table)
        checkerboard_color_intervals = [checkerboard_colors[1,i]:checkerboard_colors[2,i] for i in axes(checkerboard_colors,2)]
    end

    # allocate FermionDetMatrixMultiplier
    asym_fdm = AsymFermionDetMatrix{T,real(T)}(
        expnΔτV, coshΔτt, sinhΔτt,
        checkerboard_neighbor_table, checkerboard_perm, checkerboard_color_intervals,
        rtmp1, rtmp2, ztmp1, ztmp2
    )

    # upate FermionDetMatrixMultiplier
    update!(asym_fdm, fpi)

    return asym_fdm
end


# udpate fermion determinant matrix to reflect fermion path integral
function update!(
    fdm::AbstractFermionDetMatrix{T, E},
    fpi::FermionPathIntegral{T, E}
) where {T<:Number, E<:AbstractFloat}

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_perm) = fdm
    (; t, V, Δτ) = fpi

    # iterate over orbitals
    @views @. expnΔτV = exp(-Δτ * V')

    # imaginary-time discretization used in checkerboard approximation
    Δτ′ = isa(fdm, AsymFermionDetMatrix) ? Δτ : Δτ/2

    # iterate over hopping
    @simd for h in axes(t, 1)
        # iterate over imaginary-time slice
        for l in axes(t, 2)
            h′ = checkerboard_perm[h]
            t′ = t[h′, l]
            coshΔτt[l,h] = cosh(Δτ′ * abs(t′))
            sinhΔτt[l,h] = sign(conj(t′)) * sinh(Δτ′ * abs(t′))
        end
    end

    return nothing
end


# return matrix element type of fermion determinant matrix
eltype(fdm::AbstractFermionDetMatrix{T}) where {T} = T

# return size of fermion determinant matrix
size(fdm::AbstractFermionDetMatrix) = (length(fdm.expnΔτV), length(fdm.expnΔτV))
size(fdm::AbstractFermionDetMatrix, dim::Int) = length(fdm.expnΔτV)


# evaluate v = Mᵀ⋅M⋅v
function lmul!(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    mul!(v, fdm, v)

    return nothing
end


# evaluate v′ = Mᵀ⋅M⋅v
function mul!(
    v′::AbstractVecOrMat,
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    mul_MtM!(v′, fdm, v)

    return nothing
end


# evaluate v = Mᵀ⋅M⋅v
function lmul_MtM!(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    mul_MtM!(v, fdm, v)

    return nothing
end


# evaluate v′ = Mᵀ⋅M⋅v
function mul_MtM!(
    v′::AbstractVecOrMat,
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    tmp1, _ =  _get_tmp(fdm, v)
    mul_M!(tmp1, fdm, v)
    mul_Mt!(v′, fdm, tmp1)

    return nothing
end


# evaluate v = M⋅Mᵀ⋅v
function lmul_MMt!(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    mul_MMt!(v, fdm, v)

    return nothing
end


# evaluate v′ = M⋅Mᵀ⋅v
function mul_MMt!(
    v′::AbstractVecOrMat,
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    tmp1, _ =  _get_tmp(fdm, v)
    mul_Mt!(tmp1, fdm, v)
    mul_M!(v′, fdm, tmp1)

    return nothing
end


# evaluate v = M⋅v
function lmul_M!(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    _, tmp2 =  _get_tmp(fdm, v)
    copyto!(tmp2, v)
    mul_M!(v, fdm, tmp2)

    return nothing
end

# evaluate v′ = M⋅v
function mul_M!(
    v′::AbstractVecOrMat,
    fdm::SymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fdm
    Lτ = size(expnΔτV, 1)
    N = size(expnΔτV, 2)
    u = reshaped(v, Lτ, N)
    u′ = reshaped(v′, Lτ, N)

    # v′[l] = v[l-1]
    circshift!(u′, u, (1,0))

    # v′[l] = exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    checkerboard_lmul!(
        u′, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = true, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # v′[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    @. u′ = expnΔτV * u′

    # v′[l] = B[l]⋅v[l-1] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    checkerboard_lmul!(
        u′, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = false, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # iterate over orbitals
    for i in axes(u′, 2)
        # v′[1] = v[1] + B[1]⋅v[Lτ]
        u′[1,i] = u[1,i] + u′[1,i]
        # iterate over imaginary-time slice
        @simd for l in 2:Lτ
            # v′[l] = v[l] - B[l]⋅v[l-1]
            u′[l,i] = u[l,i] - u′[l,i]
        end
    end

    return nothing
end

# evaluate v′ = M⋅v
function mul_M!(
    v′::AbstractVecOrMat,
    fdm::AsymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fdm
    Lτ = size(expnΔτV, 1)
    N = size(expnΔτV, 2)
    u = reshaped(v, Lτ, N)
    u′ = reshaped(v′, Lτ, N)

    # v′[l] = v[l-1]
    circshift!(u′, u, (1,0))

    # v′[l] = exp(-Δτ⋅K[l])⋅v[l-1]
    checkerboard_lmul!(
        u′, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = false, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # v′[l] = B[l]⋅v[l-1] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])⋅v[l-1]
    @. u′ = expnΔτV * u′

    # iterate over orbitals
    for i in axes(u′, 2)
        # v′[1] = v[1] + B[1]⋅v[Lτ]
        u′[1,i] = u[1,i] + u′[1,i]
        # iterate over imaginary-time slice
        @simd for l in 2:Lτ
            # v′[l] = v[l] - B[l]⋅v[l-1]
            u′[l,i] = u[l,i] - u′[l,i]
        end
    end

    return nothing
end


# evaluate v = M⋅v
function lmul_Mt!(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat
)

    _, tmp2 =  _get_tmp(fdm, v)
    copyto!(tmp2, v)
    mul_Mt!(v, fdm, tmp2)

    return nothing
end


# evaluate v′ = Mᵀ⋅v
function mul_Mt!(
    v′::AbstractVecOrMat,
    fdm::SymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fdm
    Lτ = size(expnΔτV, 1)
    N = size(expnΔτV, 2)
    u = reshaped(v, Lτ, N)
    u′ = reshaped(v′, Lτ, N)

    # v′[l] = exp(-Δτ⋅K[l]/2)ᵀ⋅v[l]
    checkerboard_mul!(
        u′, u, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = true, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # v′[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l]
    @. u′ = expnΔτV * u′

    # v′[l] = Bᵀ[l]⋅v[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l]
    checkerboard_lmul!(
        u′, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = false, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # iterate over orbitals
    for i in axes(u, 2)

        # record v′[Lτ] = v[Lτ] + Bᵀ[1]⋅v[1] for l = Lτ
        vp_Lτ_i = u[Lτ,i] + u′[1,i]

        # iterate over imaginary-time slices
        @simd for l in 1:Lτ-1

            # v′[l] = v[l] - Bᵀ[l+1]⋅v[l+1] for l < Lτ
            u′[l,i] = u[l,i] - u′[l+1,i]
        end

        # apply v′[Lτ] = v[Lτ] + Bᵀ[1]⋅v[1] for l = Lτ
        u′[Lτ,i] = vp_Lτ_i
    end

    return nothing
end

# evaluate v′ = Mᵀ⋅v
function mul_Mt!(
    v′::AbstractVecOrMat,
    fdm::AsymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fdm
    Lτ = size(expnΔτV, 1)
    N = size(expnΔτV, 2)
    u = reshaped(v, Lτ, N)
    u′ = reshaped(v′, Lτ, N)

    # v′[l] = exp(-Δτ⋅V[l])⋅v[l]
    @. u′ = expnΔτV * u

    # v′[l] = Bᵀ[l]⋅v[l] = exp(-Δτ⋅K[l])ᵀ⋅exp(-Δτ⋅V[l])⋅v[l]
    checkerboard_lmul!(
        u′, checkerboard_neighbor_table,  coshΔτt, sinhΔτt,
        transposed = true, interval = 1:size(checkerboard_neighbor_table, 2)
    )

    # iterate over orbitals
    for i in axes(u, 2)

        # record v′[Lτ] = v[Lτ] + Bᵀ[1]⋅v[1] for l = Lτ
        vp_Lτ_i = u[Lτ,i] + u′[1,i]

        # iterate over imaginary-time slices
        @simd for l in 1:Lτ-1

            # v′[l] = v[l] - Bᵀ[l+1]⋅v[l+1] for l < Lτ
            u′[l,i] = u[l,i] - u′[l+1,i]
        end

        # apply v′[Lτ] = v[Lτ] + Bᵀ[1]⋅v[1] for l = Lτ
        u′[Lτ,i] = vp_Lτ_i
    end

    return nothing
end

# get temporary storage vectors of complex numbers
function _get_tmp(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat{Complex{T}}
) where {T<:AbstractFloat}

    return fdm.ztmp1, fdm.ztmp2
end

# get temporary storage vectors of real numbers
function _get_tmp(
    fdm::AbstractFermionDetMatrix,
    v::AbstractVecOrMat{T}
) where {T<:AbstractFloat}

    return fdm.rtmp1, fdm.rtmp2
end