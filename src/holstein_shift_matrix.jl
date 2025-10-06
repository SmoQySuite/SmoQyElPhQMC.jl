# calculate matrix elements Λₗₙ = (2δₗ - 1)⋅exp(+Δτ⋅(α⋅xₗₚ + α₃⋅xₗₚ³)/2)
function update_Λ!(
    Λ::AbstractMatrix{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E}
) where {T<:Number, E<:AbstractFloat}

    (; Δτ, x, holstein_parameters_up) = electron_phonon_parameters
    (; Nholstein, nholstein, α, α3, coupling_to_phonon, coupling_to_site, ph_sym_form) = holstein_parameters_up

    # initialize assuming no holstein couplings
    @views @. Λ[1, :] = 1.0
    @views @. Λ[2:end, :] = -1.0

    # if there are particle-hole symmetric form holstein couplings
    if any(ph_sym_form)
        # Number of unit cells
        Nunitcells = Nholstein ÷ nholstein
        # iterate over types of holstein coupling
        for nhol in 1:nholstein
            # if holstein coupling is shiffted
            if ph_sym_form[nhol]
                # iterate over unit cells
                for uc in 1:Nunitcells
                    # get the holstein coupling
                    coupling = (nhol-1) * Nunitcells + uc
                    # get phonon mode
                    phonon  = coupling_to_phonon[coupling]
                    # get site in lattice
                    site = coupling_to_site[coupling]
                    # get couplings
                    αc = α[coupling]
                    α3c = α3[coupling]
                    # get views into relevant arrays
                    Λ_i = @view Λ[:,site]
                    x_p = @view x[phonon,: ]
                    # update matrix
                    @. Λ_i = exp(+Δτ*(αc * x_p + α3c * x_p^3)/2) * Λ_i
                end
            end
        end
    end

    return nothing
end

# evalute |u′⟩ = Λ|u⟩
function mul_Λ!(
    u′::AbstractVecOrMat{T},
    Λ::AbstractMatrix{E},
    u::AbstractVecOrMat{T}
) where {T<:Number, E<:AbstractFloat}

    v′ = reshaped(u′, size(Λ))
    v  = reshaped(u, size(Λ))
    # length of imaginary-time axis
    Lτ = size(Λ, 1)
    # iterate of orbitals
    @inbounds for n in axes(Λ,2)
        # record the v[1,n] vector element
        v_1_n = v[1,n]
        # iterate over imaginary time-slices
        @simd for l in 1:Lτ-1
            # update vector element
            v′[l,n] = Λ[l+1,n] * v[l+1,n]
        end
        # update vector element for l = Lτ
        v′[Lτ,n] = Λ[1,n] * v_1_n
    end

    return nothing
end

# evaluate |u′⟩ = Λ⁻¹|u⟩
function ldiv_Λ!(
    u′::AbstractVecOrMat{T},
    Λ::AbstractMatrix{E},
    u::AbstractVecOrMat{T}
) where {T<:Number, E<:AbstractFloat}

    v′ = reshaped(u′, size(Λ))
    v  = reshaped(u, size(Λ))
    # length of imaginary-time axis
    Lτ = size(Λ, 1)
    # iterate of orbitals
    @inbounds for n in axes(Λ,2)
        # record the v[Lτ,n] vector element
        v_Lτ_n = v[Lτ,n]
        # iterate over imaginary time-slices
        @simd for l in Lτ:-1:2
            # update vector element
            v′[l,n] = v[l-1,n] / Λ[l,n]
        end
        # update vector element for l=1
        v′[1,n] = v_Lτ_n / Λ[1,n]
    end

    return nothing
end


# evalute |u′⟩ = Λᵀ|u⟩
function mul_Λᵀ!(
    u′::AbstractVecOrMat{T},
    Λ::AbstractMatrix{E},
    u::AbstractVecOrMat{T}
) where {T<:Number, E<:AbstractFloat}

    v′ = reshaped(u′, size(Λ))
    v  = reshaped(u, size(Λ))
    # length of imaginary-time axis
    Lτ = size(Λ, 1)
    # iterate of orbitals
    @inbounds for n in axes(Λ,2)
        # record the v[Lτ,n] vector element
        v_Lτ_n = v[Lτ,n]
        # iterate over imaginary time-slices
        @simd for l in Lτ:-1:2
            # update vector element
            v′[l,n] = Λ[l,n] * v[l-1,n]
        end
        # update vector element for l=1
        v′[1,n] = Λ[1,n] * v_Lτ_n
    end

    return nothing
end

# evaluate |u′⟩ = Λ⁻ᵀ|u⟩
function ldiv_Λᵀ!(
    u′::AbstractVecOrMat{T},
    Λ::AbstractMatrix{E},
    u::AbstractVecOrMat{T}
) where {T<:Number, E<:AbstractFloat}

    v′ = reshaped(u′, size(Λ))
    v  = reshaped(u, size(Λ))
    # length of imaginary-time axis
    Lτ = size(Λ, 1)
    # iterate of orbitals
    @inbounds for n in axes(Λ,2)
        # record the v[Lτ,n] vector element
        v_1_n = v[1,n]
        # iterate over imaginary time-slices
        @simd for l in 1:Lτ-1
            # update vector element
            v′[l,n] = v[l+1,n] / Λ[l+1,n]
        end
        # update vector element for l=1
        v′[Lτ,n] = v_1_n / Λ[1,n]
    end

    return nothing
end

# evaluate ν⋅Re[⟨u′|∂Λ/∂x|u⟩]
function mul_νRe∂Λ∂x!(
    νRe∂Λ∂x::AbstractMatrix{E},
    ν::E,
    u′::AbstractVecOrMat,
    u::AbstractVecOrMat,
    Λ::AbstractMatrix{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E}
) where {T<:Number, E<:AbstractFloat}

    (; Lτ, Δτ, x, holstein_parameters_up) = electron_phonon_parameters
    (; Nholstein, nholstein, α, α3, coupling_to_phonon, coupling_to_site, ph_sym_form) = holstein_parameters_up
    v′ = reshaped(u′, size(Λ))
    v  = reshaped(u, size(Λ))

    # if there are particle-hole symmetric form holstein couplings
    if any(ph_sym_form)
        # number of unit cells in lattice
        Nunitcells = Nholstein ÷ nholstein
        # iterate over types of holstein coupling
        @inbounds for nhol in 1:nholstein
            # if holstein coupling is shiffted
            if ph_sym_form[nhol]
                # iterate over unit cells
                for uc in 1:Nunitcells
                    # get the holstein coupling
                    coupling = (nhol-1) * Nunitcells + uc
                    # get couplings
                    αc = α[coupling]
                    α3c = α3[coupling]
                    # get phonon
                    phonon = coupling_to_phonon[coupling]
                    # get the site
                    site = coupling_to_site[coupling]
                    # iterate over imaginary time slices
                    @simd for l in axes(Λ, 1)                    
                        # calculate c⋅Re[⟨v′|∂Λ/∂x|v⟩]
                        Δτ∂Λ∂x = Δτ * (αc + 3*α3c * x[phonon,l]^2)/2 * Λ[l,site]
                        νRe∂Λ∂x[phonon,l] += ν * real( conj(v′[mod1(l-1,Lτ),site]) * Δτ∂Λ∂x * v[l,site] )
                    end
                end
            end
        end
    end

    return nothing
end