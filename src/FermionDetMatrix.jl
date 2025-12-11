@doc raw"""
    FermionDetMatrix{T<:Number, E<:AbstractFloat}

A abstract type to represent fermion determinant matrix
```math
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
abstract type FermionDetMatrix{T<:Number, E<:AbstractFloat} end

@doc raw"""
    SymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: FermionDetMatrix{T,E}

A type to represent fermion determinant matrix
```math
M = \left(\begin{array}{ccccc}
    I &  &  &  & B_{0}\\
    -B_{1} & I\\
    & -B_{2} & \ddots\\
    &  & \ddots & \ddots\\
    &  &  & -B_{L_{\tau}-1} & I
\end{array}\right),
```
where
```math
B_l = \left[ e^{-\Delta\tau K_l/2} \right]^\dagger e^{-\Delta\tau V_l} e^{-\Delta\tau K_l/2}
```
are Hermitian (symmetric if real) propagator matrices for imaginary-time slice ``\tau = \Delta\tau \cdot l`` given
an inverse temperature ``\beta = \Delta\tau \cdot L_\tau``. A Fermion determinant matrix ``M``
will be ``N L_\tau \times N L_\tau``, where each propagator matrix ``B_l`` is ``N \times N``,
where ``N`` is the number of orbitals in the lattice. Here the matrix ``e^{-\Delta\tau K_l/2}`` is 
approximated using the non-hermitian checkerboard approximation.
"""
struct SymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: FermionDetMatrix{T,E}

    expnΔτV::Matrix{E}
    coshΔτt::Matrix{T}
    sinhΔτt::Matrix{T}
    checkerboard_neighbor_table::Matrix{Int}
    checkerboard_perm::Vector{Int}
    checkerboard_colors::Vector{UnitRange{Int}}
    cgs::ConjugateGradientSolver{Complex{E}, E}
    tmp1::Matrix{Complex{E}}
    tmp2::Matrix{Complex{E}}
end

@doc raw"""
    SymFermionDetMatrix(
        fermion_path_integral::FermionPathIntegral{T, E};
        maxiter::Int = (fermion_path_integral.N * fermion_path_integral.Lτ),
        tol::E = 1e-6
    ) where {T<:Number, E<:AbstractFloat}

Initialize an instance of the [`SymFermionDetMatrix`](@ref) type.
"""
function SymFermionDetMatrix(
    fermion_path_integral::FermionPathIntegral{T, E};
    maxiter::Int = (fermion_path_integral.N * fermion_path_integral.Lτ),
    tol::E = 1e-6
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, t, V, N, β, Δτ, Lτ) = fermion_path_integral

    # get number of hoppings
    Nh = size(t, 1)

    # allocate arrays
    expnΔτV = zeros(real(T), Lτ, N)
    coshΔτt = zeros(T, Lτ, Nh)
    sinhΔτt = zeros(T, Lτ, Nh)

    # allocate temorary storage vector
    tmp1 = zeros(Complex{real(T)}, Lτ, N)
    tmp2 = zeros(Complex{real(T)}, Lτ, N)

    # initialize conjugate gradient solver
    cgs = ConjugateGradientSolver(tmp1, maxiter = maxiter, tol = tol)

    # construct checkerboard decomposition
    if iszero(Nh)
        checkerboard_neighbor_table = Matrix{Int}(undef,(2,0))
        checkerboard_perm = Int[]
        checkerboard_color_intervals = UnitRange{Int}[]
    else
        checkerboard_neighbor_table = copy(neighbor_table)
        checkerboard_perm, checkerboard_colors = checkerboard_decomposition!(checkerboard_neighbor_table)
        checkerboard_color_intervals = [checkerboard_colors[1,i]:checkerboard_colors[2,i] for i in axes(checkerboard_colors,2)]
    end

    # allocate FermionDetMatrixMultiplier
    sym_fdm = SymFermionDetMatrix{T,real(T)}(
        expnΔτV, coshΔτt, sinhΔτt,
        checkerboard_neighbor_table, checkerboard_perm, checkerboard_color_intervals,
        cgs, tmp1, tmp2
    )

    # upate FermionDetMatrixMultiplier
    update!(sym_fdm, fermion_path_integral)

    return sym_fdm
end


@doc raw"""
    AsymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: FermionDetMatrix{T, E}

A type to represent fermion determinant matrix
```math
M = \left(\begin{array}{ccccc}
    I &  &  &  & B_{0}\\
    -B_{1} & I\\
    & -B_{2} & \ddots\\
    &  & \ddots & \ddots\\
    &  &  & -B_{L_{\tau}-1} & I
\end{array}\right),
```
where
```math
B_l = e^{-\Delta\tau V_l} e^{-\Delta\tau K_l}
```
are Hermitian (symmetric if real) propagator matrices for imaginary-time slice ``\tau = \Delta\tau \cdot l`` given
an inverse temperature ``\beta = \Delta\tau \cdot L_\tau``. A Fermion determinant matrix ``M``
will be ``N L_\tau \times N L_\tau``, where each propagator matrix ``B_l`` is ``N \times N``,
where ``N`` is the number of orbitals in the lattice. Note that ``e^{-\Delta\tau K_l}`` is
represented using the non-hermitian checkerboard approximation.
"""
struct AsymFermionDetMatrix{T<:Number, E<:AbstractFloat} <: FermionDetMatrix{T,E}

    expnΔτV::Matrix{E}
    coshΔτt::Matrix{T}
    sinhΔτt::Matrix{T}
    checkerboard_neighbor_table::Matrix{Int}
    checkerboard_perm::Vector{Int}
    checkerboard_colors::Vector{UnitRange{Int}}
    cgs::ConjugateGradientSolver{Complex{E}, E}
    tmp1::Matrix{Complex{E}}
    tmp2::Matrix{Complex{E}}
end

@doc raw"""
    AsymFermionDetMatrix(
        fermion_path_integral::FermionPathIntegral{T, E};
        maxiter::Int = (fermion_path_integral.N * fermion_path_integral.Lτ),
        tol::E = 1e-6
    ) where {T<:Number, E<:AbstractFloat}

Initialize an instance of the [`AsymFermionDetMatrix`](@ref) type.
"""
function AsymFermionDetMatrix(
    fermion_path_integral::FermionPathIntegral{T, E};
    maxiter::Int = (fermion_path_integral.N * fermion_path_integral.Lτ),
    tol::E = 1e-6
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, t, N, Lτ) = fermion_path_integral

    # get number of hoppings
    Nh = size(t, 1)

    # allocate arrays
    expnΔτV = zeros(real(T), Lτ, N)
    coshΔτt = zeros(T, Lτ, Nh)
    sinhΔτt = zeros(T, Lτ, Nh)

    # allocate temorary storage vector
    tmp1 = zeros(Complex{real(T)}, Lτ, N)
    tmp2 = zeros(Complex{real(T)}, Lτ, N)

    # initialize conjugate gradient solver
    cgs = ConjugateGradientSolver(tmp1, maxiter = maxiter, tol = tol)

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
        cgs, tmp1, tmp2
    )

    # upate FermionDetMatrixMultiplier
    update!(asym_fdm, fermion_path_integral)

    return asym_fdm
end


# update fermion determinant matrix to reflect fermion path integral
function update!(
    fermion_det_matrix::FermionDetMatrix{T, E},
    fermion_path_integral::FermionPathIntegral{T, E}
) where {T<:Number, E<:AbstractFloat}

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_perm) = fermion_det_matrix
    (; t, V, Δτ) = fermion_path_integral

    # iterate over orbitals
    @views @. expnΔτV = exp(-Δτ * V')

    # imaginary-time discretization used in checkerboard approximation
    Δτ′ = isa(fermion_det_matrix, AsymFermionDetMatrix) ? Δτ : Δτ/2

    # iterate over hopping
    @inbounds for h in axes(t, 1)
        h′ = checkerboard_perm[h]
        t_h′ = @view t[h′, :]
        # iterate over imaginary-time slice
        @simd for l in axes(t, 2)
            t′ = t_h′[l]
            Δτt_abs = Δτ′ * abs(t′)
            coshΔτt[l,h] = cosh(Δτt_abs)
            sinhΔτt[l,h] = sign(conj(t′)) * sinh(Δτt_abs)
        end
    end

    return nothing
end


# return matrix element type of fermion determinant matrix
eltype(fermion_det_matrix::FermionDetMatrix{T}) where {T} = T

# return size of fermion determinant matrix
size(fermion_det_matrix::FermionDetMatrix) = (length(fermion_det_matrix.expnΔτV), length(fermion_det_matrix.expnΔτV))
size(fermion_det_matrix::FermionDetMatrix, dim::Int) = length(fermion_det_matrix.expnΔτV)


# evaluate v′ = [Mᵀ⋅M]⁻¹⋅v
function ldiv!(
    v′::AbstractVecOrMat{Complex{E}},
    fermion_det_matrix::FermionDetMatrix{T, E},
    v::AbstractVecOrMat{Complex{E}};
    preconditioner = I,
    rng::AbstractRNG = Random.default_rng(),
    maxiter::Int = fermion_det_matrix.cgs.maxiter,
    tol::E = fermion_det_matrix.cgs.tol
) where {T<:Number, E<:AbstractFloat}

    (; cgs) = fermion_det_matrix
    update_preconditioner!(preconditioner, fermion_det_matrix, rng)
    iters, ϵ = cg_solve!(
        v′, fermion_det_matrix, v, cgs, preconditioner,
        maxiter = maxiter,
        tol = tol
    )

    return iters, ϵ
end

# evaluate v = [Mᵀ⋅M]⁻¹⋅v
function ldiv!(
    fermion_det_matrix::FermionDetMatrix{T, E},
    v::AbstractVecOrMat{Complex{E}};
    preconditioner = I,
    maxiter::Int = fermion_det_matrix.cgs.maxiter,
    tol::E = fermion_det_matrix.cgs.tol,
    rng::AbstractRNG = Random.default_rng()
) where {T<:Number, E<:AbstractFloat}

    iters, ϵ = ldiv!(
        v, fermion_det_matrix, v,
        preconditioner = preconditioner,
        rng = rng,
        maxiter = maxiter,
        tol = tol
    )

    return iters, ϵ
end


# evaluate v = Mᵀ⋅M⋅v
function lmul!(
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    mul!(v, fermion_det_matrix, v)

    return nothing
end


# evaluate v′ = Mᵀ⋅M⋅v
function mul!(
    v′::AbstractVecOrMat,
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    mul_MtM!(v′, fermion_det_matrix, v)

    return nothing
end


# evaluate v = Mᵀ⋅M⋅v
function lmul_MtM!(
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    mul_MtM!(v, fermion_det_matrix, v)

    return nothing
end


# evaluate v′ = Mᵀ⋅M⋅v
function mul_MtM!(
    v′::AbstractVecOrMat,
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    (; tmp1) = fermion_det_matrix
    mul_M!(tmp1, fermion_det_matrix, v)
    mul_Mt!(v′, fermion_det_matrix, tmp1)

    return nothing
end


# evaluate v = M⋅Mᵀ⋅v
function lmul_MMt!(
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    
    mul_MMt!(v, fermion_det_matrix, v)

    return nothing
end


# evaluate v′ = M⋅Mᵀ⋅v
function mul_MMt!(
    v′::AbstractVecOrMat,
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    (; tmp1) = fermion_det_matrix
    mul_Mt!(tmp1, fermion_det_matrix, v)
    mul_M!(v′, fermion_det_matrix, tmp1)

    return nothing
end


# evaluate v = M⋅v
function lmul_M!(
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    (; tmp2) = fermion_det_matrix
    copyto!(tmp2, v)
    mul_M!(v, fermion_det_matrix, tmp2)

    return nothing
end

# evaluate v′ = M⋅v
function mul_M!(
    v′::AbstractVecOrMat,
    fermion_det_matrix::SymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fermion_det_matrix
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
    @inbounds for i in axes(u′, 2)
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
    fermion_det_matrix::AsymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fermion_det_matrix
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
    @inbounds for i in axes(u′, 2)
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
    fermion_det_matrix::FermionDetMatrix,
    v::AbstractVecOrMat
)

    (; tmp2) = fermion_det_matrix
    copyto!(tmp2, v)
    mul_Mt!(v, fermion_det_matrix, tmp2)

    return nothing
end


# evaluate v′ = Mᵀ⋅v
function mul_Mt!(
    v′::AbstractVecOrMat,
    fermion_det_matrix::SymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fermion_det_matrix
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
    @inbounds for i in axes(u, 2)
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
    fermion_det_matrix::AsymFermionDetMatrix,
    v::AbstractVecOrMat
)

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table) = fermion_det_matrix
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
    @inbounds for i in axes(u, 2)
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