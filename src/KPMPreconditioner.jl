# abstract type to represent KPM expansion
abstract type AbstractKPMPreconditioner{T<:Number, E<:AbstractFloat} end

# KPM preconditioner when fermion determinant matrix is defined using symmetric propagator matrices
mutable struct SymKPMPreconditioner{T, E, Tfft, Tifft} <: AbstractKPMPreconditioner{T, E}

    # whether preconditioner is active or not
    active::Bool
    # relative buffer applied to eigevalue bounds calculated by Lanczos
    rbuf::E
    # number of lanczos iterations used to approximate eigenvalue bounds
    n::Int
    # controls maximum order of kpm expansion
    a1::E
    # controls minimum order of kpm expansion
    a2::E
    # average propagator matrix
    B̄::SymChkbrdPropagator{T, E}
    # type to fourier transform vectors back and forth between τ and ωₙ
    U::FourierTransformer{E, Tfft, Tifft}
    # phase associated with each frequency
    ϕs::Vector{E}
    # eivenvalue bounds of B̄
    bounds::Tuple{E, E}
    # order of KPM expansion for each frequency
    order::Vector{Int}
    # KPM expansion coefficients for each frequency
    coefs::Vector{Vector{E}}
    # buffer used to calculate KPM coefficients
    buf::Vector{E}
    # temporary storage space for KPM coefficient calculation
    tmp::Matrix{Complex{E}}
    # vectors for performing Lanczos
    α_lanczos::Vector{E}
    # vectors for performing Lanczos
    β_lanczos::Vector{E}
    # temporar storage for Lanczos
    tmp_lanczos::Matrix{T}
    # temprorary storage vectors for ldiv!
    v::Matrix{Complex{E}}
    # temporary storage vectors for ldiv!
    v′::Matrix{Complex{E}}
end

# KPM preconditioner when fermion determinant matrix is defined using asymmetric propagator matrices
mutable struct AsymKPMPreconditioner{T, E, Tfft, Tifft} <: AbstractKPMPreconditioner{T, E}

    # whether preconditioner is active or not
    active::Bool
    # relative buffer applied to eigevalue bounds calculated by Lanczos
    rbuf::E
    # number of lanczos iterations used to approximate eigenvalue bounds
    n::Int
    # controls maximum order of kpm expansion
    a1::E
    # controls minimum order of kpm expansion
    a2::E
    # average propagator matrix
    B̄::AsymChkbrdPropagator{T, E}
    # type to fourier transform vectors back and forth between τ and ωₙ
    U::FourierTransformer{E, Tfft, Tifft}
    # phase associated with each frequency
    ϕs::Vector{E}
    # eivenvalue bounds of B̄
    bounds::Tuple{E, E}
    # order of KPM expansion for each frequency
    order::Vector{Int}
    # KPM expansion coefficients for each frequency
    coefs::Vector{Vector{Complex{E}}}
    # buffer used to calculate KPM coefficients
    buf::Vector{E}
    # temporary storage space for KPM coefficient calculation
    tmp::Matrix{Complex{E}}
    # vectors for performing Lanczos
    α_lanczos::Vector{E}
    # vectors for performing Lanczos
    β_lanczos::Vector{E}
    # temporar storage for Lanczos
    tmp_lanczos::Matrix{T}
    # temprorary storage vectors for ldiv!
    v::Matrix{Complex{E}}
    # temporary storage vectors for ldiv!
    v′::Matrix{Complex{E}}
end

# initialize a KPM preconditioner
function KPMPreconditioner(
    fdm::AbstractFermionDetMatrix{T},
    rng::AbstractRNG;
    rbuf::E = 0.10,
    n::Int = 20,
    a1::E = 1.0,
    a2::E = 1.0
) where {T<:Number, E<:AbstractFloat}

    (; expnΔτV, coshΔτt, sinhΔτt) = fdm
    (; checkerboard_neighbor_table, checkerboard_perm, checkerboard_colors) = fdm
    (Lτ, N) = size(expnΔτV)

    # initialize temporaray storage vectors
    v = zeros(Complex{E}, Lτ, N)
    v′ = zeros(Complex{E}, N, Lτ)

    # initialize Fourier transformer
    U = FourierTransformer(E, Lτ, N)

    # initialize complex phases
    ϕs = [2π/Lτ*(ω+1/2) for ω in 0:(Lτ-1)]

    # half the frequencies
    Lτo2 = cld(Lτ, 2)

    # buffer for calculating coefficient
    buf = E[]

    # temporary storage for KPM matrix-vector multiply
    tmp = zeros(Complex{E}, N, 3)

    # initialize bounds
    bounds = (0.0, 0.0)

    # vectors for performing Lanczos
    α_lanczos = zeros(E, n)
    β_lanczos = zeros(E, n-1)
    tmp_lanczos = zeros(T, N, 6)

    # initialize checkerboard matrix
    Γ = CheckerboardMatrix{T}(
        false, false, N, size(checkerboard_neighbor_table, 2), length(checkerboard_colors),
        checkerboard_neighbor_table, vec(mean(coshΔτt, dims=1)), vec(mean(sinhΔτt, dims=1)),
        checkerboard_perm, invperm(checkerboard_perm), hcat([[first(r),last(r)] for r in checkerboard_colors]...)
    )

    if isa(fdm, SymFermionDetMatrix)

        # get half the frequencies
        Lτo2 = cld(Lτ, 2)

        # initialize order of KPM expansion for each frquency
        order = zeros(Int, Lτo2)

        # initialize vector of vector to contain coefficients
        coefs = [E[] for _ in 1:Lτo2]

        # initialize symmetric checkerboard propagator
        B̄ = SymChkbrdPropagator(vec(mean(expnΔτV, dims=1)), Γ)

        # initialize preconditioner
        Pkpm = SymKPMPreconditioner(false, rbuf, n, 2*a1, a2, B̄, U, ϕs, bounds, order, coefs, buf, tmp, α_lanczos, β_lanczos, tmp_lanczos, v, v′)

    else

        # initialize order of KPM expansion for each frquency
        order = zeros(Int, Lτ)

        # initialize vector of vector to contain coefficients
        coefs = [Complex{E}[] for _ in 1:Lτ]

        # initialize asymmetric checkerboard propagator
        B̄ = AsymChkbrdPropagator(vec(mean(expnΔτV, dims=1)), Γ)

        # initialize preconditioner
        Pkpm = AsymKPMPreconditioner(false, rbuf, n, a1, a2, B̄, U, ϕs, bounds, order, coefs, buf, tmp, α_lanczos, β_lanczos, tmp_lanczos, v, v′)
    end

    # update KPM preconditioner
    update_preconditioner!(Pkpm, fdm, rng)

    return Pkpm
end


# evaluate u′ = P⁻¹⋅u where P is the SymKPMPreconditioner u and u′ are real vectors
function ldiv!(
    u′::AbstractVecOrMat{T},
    Pkpm::SymKPMPreconditioner,
    u::AbstractVecOrMat{T}
) where {T<:AbstractFloat}

    (; v, v′, B̄, U, bounds, tmp, order) = Pkpm
    (; Lτ, N) = U

    # reshaped in case vectors as inputs
    u  = reshaped(u, Lτ, N)
    u′ = reshaped(u′, Lτ, N)

    # get half the frequencies
    Lτo2 = cld(Lτ, 2)

    # check if preconditioner is active
    if Pkpm.active

        # transform τ → ωₙ
        mul!(v, U, u)

        # v′[r,ω] = v[ω,r]
        transpose!(v′, v)

        # iterate over frequncies
        for n in 1:Lτo2

            # get relevant vector
            vₙ = @view v′[:,n]

            # get coefficients
            coefs = Pkpm.coefs[n]

            # if expansion order is greater than 1
            if order[n] > 1

                # evaluate (M̃[ωₙ,ωₙ]ᵀ⋅M̃[ωₙ,ωₙ])⁻¹⋅vₙ
                kpm_lmul!(B̄, coefs, vₙ, bounds, tmp)
            else

                # apply single order expansion
                @. vₙ = coefs * vₙ
            end

            # account for complex conjugacy
            @views @. v′[:, Lτ-n+1] = conj(vₙ)
        end

        # v[ω,r] = v′[r,ω]
        transpose!(v, v′)

        # transform ωₙ → τ
        ldiv!(U, v)

        # record result
        @. u′ = real(v)
    else
    
        # copy u to u′
        copyto!(u′, u)
    end

    return nothing
end

# evaluate u′ = P⁻¹⋅u where P is the SymKPMPreconditioner u and u′ are complex vectors
function ldiv!(
    u′::AbstractVecOrMat{Complex{T}},
    Pkpm::SymKPMPreconditioner,
    u::AbstractVecOrMat{Complex{T}}
) where {T<:AbstractFloat}

    (; v, v′, B̄, U, bounds, tmp, order) = Pkpm
    (; Lτ, N) = U

    # reshaped in case vectors as inputs
    u  = reshaped(u, Lτ, N)
    u′ = reshaped(u′, Lτ, N)

    # get half the frequencies
    Lτo2 = cld(Lτ, 2)

    # check if preconditioner is active
    if Pkpm.active

        # transform τ → ωₙ
        mul!(v, U, u)

        # v′[r,ω] = v[ω,r]
        transpose!(v′, v)

        # iterate over frequncies
        for n in 1:Lτ

            # get relevant vector
            vₙ = @view v′[:,n]

            # get coefficients
            n′ = n > Lτo2 ? Lτ-n+1 : n
            coefs = Pkpm.coefs[n′]

            # if expansion order is greater than 1
            if order[n′] > 1

                # evaluate (M̃[ωₙ,ωₙ]ᵀ⋅M̃[ωₙ,ωₙ])⁻¹⋅vₙ
                kpm_lmul!(B̄, coefs, vₙ, bounds, tmp)
            else

                # apply single order expansion
                @. vₙ = coefs * vₙ
            end
        end

        # v[ω,r] = v′[r,ω]
        transpose!(u′, v′)

        # transform ωₙ → τ
        ldiv!(U, u′)
    else
    
        # copy u to u′
        copyto!(u′, u)
    end

    return nothing
end

# evaluate u′ = P⁻¹⋅u where P is the AsymKPMPreconditioner  u and u′ are real vectors
function ldiv!(
    u′::AbstractVecOrMat{T},
    Pkpm::AsymKPMPreconditioner,
    u::AbstractVecOrMat{T}
) where {T<:AbstractFloat}

    (; v, v′, B̄, U, bounds, tmp, order) = Pkpm
    (; Lτ, N) = U

    # reshaped in case vectors as inputs
    u  = reshaped(u, Lτ, N)
    u′ = reshaped(u′, Lτ, N)

    # get half the frequencies
    Lτo2 = cld(Lτ, 2)

    # check if preconditioner is active
    if Pkpm.active

        # transform τ → ωₙ
        mul!(v, U, u)

        # v′[r,ω] = v[ω,r]
        transpose!(v′, v)

        # iterate over frequncies
        for n in 1:Lτo2

            # get relevant vector
            vₙ = @view v′[:,n]

            # if expansion order is greater than 1
            if order[n] > 1

                # get coefficients
                coefs_conj = Pkpm.coefs[Lτ-n+1]
                coefs = Pkpm.coefs[n]

                # evaluate M̃[ωₙ,ωₙ]⁻ᵀ⋅vₙ
                kpm_lmul!(B̄, coefs_conj, vₙ, bounds, tmp)

                # evaluate M̃[ωₙ,ωₙ]⁻¹⋅(M̃[ωₙ,ωₙ]⁻ᵀ⋅vₙ) = (M̃[ωₙ,ωₙ]ᵀ⋅M̃[ωₙ,ωₙ])⁻¹⋅vₙ
                kpm_lmul!(B̄, coefs, vₙ, bounds, tmp)
            else

                coefs = Pkpm.coefs[n]
                @. vₙ = abs2(coefs[1]) * vₙ
            end

            # account for complex conjugacy
            @views @. v′[:, Lτ-n+1] = conj(vₙ)
        end

        # v[ω,r] = v′[r,ω]
        transpose!(v, v′)

        # transform ωₙ → τ
        ldiv!(U, v)

        # record result
        @. u′ = real(v)
    else
    
        # copy u to u′
        copyto!(u′, u)
    end

    return nothing
end

# evaluate u′ = P⁻¹⋅u where P is the AsymKPMPreconditioner  u and u′ are complex vectors
function ldiv!(
    u′::AbstractVecOrMat{Complex{T}},
    Pkpm::AsymKPMPreconditioner,
    u::AbstractVecOrMat{Complex{T}}
) where {T<:AbstractFloat}

    (; v, v′, B̄, U, bounds, tmp, order) = Pkpm
    (; Lτ, N) = U

    # reshaped in case vectors as inputs
    u  = reshaped(u, Lτ, N)
    u′ = reshaped(u′, Lτ, N)

    # get half the frequencies
    Lτo2 = cld(Lτ, 2)

    # check if preconditioner is active
    if Pkpm.active

        # transform τ → ωₙ
        mul!(v, U, u)

        # v′[r,ω] = v[ω,r]
        transpose!(v′, v)

        # iterate over frequncies
        for n in 1:Lτ

            # get relevant vector
            vₙ = @view v′[:,n]

            # if expansion order is greater than 1
            if order[n] > 1

                # get coefficients
                coefs_conj = Pkpm.coefs[Lτ-n+1]
                coefs = Pkpm.coefs[n]

                # evaluate M̃[ωₙ,ωₙ]⁻ᵀ⋅vₙ
                kpm_lmul!(B̄, coefs_conj, vₙ, bounds, tmp)

                # evaluate M̃[ωₙ,ωₙ]⁻¹⋅(M̃[ωₙ,ωₙ]⁻ᵀ⋅vₙ) = (M̃[ωₙ,ωₙ]ᵀ⋅M̃[ωₙ,ωₙ])⁻¹⋅vₙ
                kpm_lmul!(B̄, coefs, vₙ, bounds, tmp)
            else

                coefs = Pkpm.coefs[n]
                @. vₙ = abs2(coefs[1]) * vₙ
            end
        end

        # v[ω,r] = v′[r,ω]
        transpose!(u′, v′)

        # transform ωₙ → τ
        ldiv!(U, u′)
    else
    
        # copy u to u′
        copyto!(u′, u)
    end

    return nothing
end


# update KPM preconditioner to reflect fermion determinant matrix
function update_preconditioner!(
    Pkpm::AbstractKPMPreconditioner,
    fdm::AbstractFermionDetMatrix,
    rng::AbstractRNG
)

    (; B̄, rbuf) = Pkpm

    # update B̄ propagator matrix
    update_B̄!(B̄, fdm)

    # calculate eigenbounds
    ϵ_min_new, ϵ_max_new = calculate_bounds!(Pkpm, rng)

    # adjust eigenbounds using buffer
    ϵ_min_new = (1-rbuf)*ϵ_min_new
    ϵ_max_new = (1+rbuf)*ϵ_max_new

    # check if reasonable eigenbounds were found
    if (0.0 < ϵ_min_new < 1.0) && (1.0 < ϵ_max_new < 2.0)

        # activate the preconditioner
        Pkpm.active = true

        # get current bounds
        ϵ_min, ϵ_max = Pkpm.bounds

        # check if current bounds are no longer accurate to within the buffer tolerance
        if !isapprox(ϵ_min, ϵ_min_new, rtol = rbuf/2) || !isapprox(ϵ_max, ϵ_max_new, rtol = rbuf/2)

            # update bounds
            Pkpm.bounds = (ϵ_min_new, ϵ_max_new)

            # update KPM expansion order and coefficients
            update_kpm_expansions!(Pkpm)
        end
    else

        # deactivate the preconditioner
        Pkpm.active = false
    end

    return nothing
end

# default update! preconditioner method does nothing
update_preconditioner!(P, ignore...) = nothing


# update B̄ propagator matrix
function update_B̄!(B̄::SymChkbrdPropagator, fdm::SymFermionDetMatrix)

    mean!(B̄.expmΔτV, fdm.expnΔτV')
    mean!(B̄.expmΔτKo2.coshΔτt, fdm.coshΔτt')
    mean!(B̄.expmΔτKo2.sinhΔτt, fdm.sinhΔτt')

    return nothing
end

# update B̄ propagator matrix
function update_B̄!(B̄::AsymChkbrdPropagator, fdm::AsymFermionDetMatrix)

    mean!(B̄.expmΔτV, fdm.expnΔτV')
    mean!(B̄.expmΔτK.coshΔτt, fdm.coshΔτt')
    mean!(B̄.expmΔτK.sinhΔτt, fdm.sinhΔτt')

    return nothing
end


# updates eigenbounds of B̄ using Lanczos
function calculate_bounds!(
    Pkpm::SymKPMPreconditioner,
    rng::AbstractRNG
)

    (; B̄, α_lanczos, β_lanczos, tmp_lanczos) = Pkpm

    v = @view tmp_lanczos[:,6]
    tmp = @view tmp_lanczos[:,1:5]
    randn!(rng, v)
    symtridiag = lanczos!(α_lanczos, β_lanczos, v, B̄, tmp = tmp, rng = rng)
    ϵ_min, ϵ_max = eigmin(symtridiag), eigmax(symtridiag)

    return ϵ_min, ϵ_max
end

# updates eigenbounds of B̄ using Lanczos
function calculate_bounds!(
    Pkpm::AsymKPMPreconditioner,
    rng::AbstractRNG
)


    (; B̄, α_lanczos, β_lanczos, tmp_lanczos) = Pkpm

    v = @view tmp_lanczos[:,6]
    tmp = @view tmp_lanczos[:,1:5]
    randn!(rng, v)
    f! = (x, y) -> mul_B̄ᵀB̄!(x, B̄, y)
    symtridiag = lanczos!(α_lanczos, β_lanczos, v, f!, tmp = tmp, rng = rng)
    ϵ_min, ϵ_max = sqrt(eigmin(symtridiag)), sqrt(eigmax(symtridiag))

    return ϵ_min, ϵ_max
end

# evaluate y = B̄ᵀ⋅B̄⋅x
function mul_B̄ᵀB̄!(
    y::AbstractVector,
    B̄::AsymChkbrdPropagator,
    x::AbstractVector
)

    (; expmΔτV, expmΔτK) = B̄
    expmΔτKᵀ = adjoint(expmΔτK)
    # y = exp(-Δτ⋅K)⋅x
    mul!(y, expmΔτK, x)
    # y = exp(-2⋅Δτ⋅V)⋅exp(-Δτ⋅K)⋅x
    @. y = abs2(expmΔτV) * y
    # y = exp(-Δτ⋅K)ᵀ⋅exp(-2⋅Δτ⋅V)⋅exp(-Δτ⋅K)⋅x
    #   = [exp(-Δτ⋅V)⋅exp(-Δτ⋅K)]ᵀ⋅[exp(-Δτ⋅V)⋅exp(-Δτ⋅K)]⋅x
    #   = B̄ᵀ⋅B̄⋅x
    lmul!(expmΔτKᵀ, y)

    return nothing
end

# update KPM expansion order and coefficients
function update_kpm_expansions!(
    Pkpm::AbstractKPMPreconditioner
)

    # update KPM expansion order
    update_kpm_expansion_order!(Pkpm)

    # update KPM expansion coefficients
    update_kpm_expansion_coefs!(Pkpm)

    return nothing
end

# update KPM expansion order
function update_kpm_expansion_order!(
    Pkpm::AbstractKPMPreconditioner
)

    (; a1, a2, bounds, ϕs, order, coefs, buf) = Pkpm

    # get eigenvalue bounds
    (ϵ_min, ϵ_max) = bounds

    # iterate over frequencies
    for l in eachindex(coefs)

        # calculate new expansion order
        ϕ = ϕs[l]
        ϕ = ϕ > π ? 2π - ϕ : ϕ
        n = max(1, floor(Int, (ϵ_max - ϵ_min)*(a1/ϕ + a2)))

        # check if expansion or changed
        if n ≠ order[l]

            # record new expansion order
            order[l] = n

            # resize the vector of expansion coefficients
            resize!(coefs[l], n)
        end
    end

    # resize the buffer
    n_max = maximum(order)
    if length(buf) ≠ 2*n_max
        resize!(buf, 2*n_max)
    end

    return nothing
end

# update KPM expansion coefficients assuming symmetric propagator matrices
function update_kpm_expansion_coefs!(
    Pkpm::SymKPMPreconditioner
)

    (; bounds, ϕs, order, coefs, buf) = Pkpm
    Lτ = length(ϕs)
    Lτo2 = cld(Lτ, 2)

    # iterate over frequencies
    for l in 1:Lτo2

        # get order of exapnsion
        n = order[l]

        # get appropriately sized buffer
        buf_2n = @view buf[1:2n]

        # calculate the real part of expansion coefficients
        kpm_coefs!(coefs[l], b -> f_B̄_sym(b, ϕs[l]), bounds, buf_2n)
    end

    return nothing
end

# update KPM expansion coefficients
function update_kpm_expansion_coefs!(
    Pkpm::AsymKPMPreconditioner
)

    (; bounds, ϕs, order, coefs, buf) = Pkpm
    Lτ = length(ϕs)
    Lτo2 = cld(Lτ, 2)

    # iterate over frequencies
    for l in 1:Lτo2

        # get order of exapnsion
        n = order[l]

        # get appropriately sized buffer
        buf_2n = @view buf[1:2n]

        # array to contain real or imaginary computed coefficients
        coefs_n = @view buf[1:n]

        # get relevant phase factor
        ϕ = ϕs[l]

        # calculate the real part of expansion coefficients
        kpm_coefs!(coefs_n, b -> real(f_B̄_asym(b , ϕ)), bounds, buf_2n)
        @. coefs[l] = coefs_n

        # calculate the imaginary part of expansion coefficients
        kpm_coefs!(coefs_n, b -> imag(f_B̄_asym(b , ϕ)), bounds, buf_2n)
        @. coefs[l] += 1im * coefs_n

        # account for complex conjugacy
        @. coefs[Lτ-l+1] = conj(coefs[l])
    end

    return nothing
end


# scalar function to approximate using chebyshev expansion for
# symmetric checkerboard propagator
f_B̄_sym(b, ϕ) = inv(abs2(b) - 2*b*cos(ϕ) + 1)

# scalar function to approximate using chebyshev expansion for
# asymmetric checkerboard propagator
f_B̄_asym(b, ϕ) = inv(1.0 - exp(-im*ϕ) * b)