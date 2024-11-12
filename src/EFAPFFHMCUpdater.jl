struct EFAPFFHMCUpdater{T<:AbstractFloat, PFFT, PIFFT}

    Nt::Int
    Δt::T
    δ::T
    x0::Matrix{T}
    p::Matrix{T}
    ∂S∂x::Matrix{T}
    Λ::Matrix{T}
    Φ::Matrix{Complex{T}}
    u::Matrix{Complex{T}}
    u′::Matrix{Complex{T}}
    u″::Matrix{Complex{T}}
    efa::SmoQyDQMC.ExactFourierAccelerator{T, PFFT, PIFFT}
end

function EFAPFFHMCUpdater(;
    # KEYWORD ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T},
    fermion_det_matrix::AbstractFermionDetMatrix{T},
    Nt::Int,
    Δt::E,
    η::E = 0.0,
    δ::E = 0.05
) where {T<:Number, E<:AbstractFloat}

    (; β, Δτ, phonon_parameters, x) = electron_phonon_parameters
    (; Ω, M) = phonon_parameters

    # number of phonon modes and imaginary-time slices
    Nph, Lτ = size(x)

    # number of orbitals in lattice
    Norbitals_Lτ = size(fermion_det_matrix, 1)
    Norbitals = Norbitals_Lτ ÷ Lτ

    # allocate arrays
    x0 = zeros(E, Nph, Lτ)
    p = zeros(E, Nph, Lτ)
    ∂S∂x = zeros(E, Nph, Lτ)
    Φ = zeros(Complex{E}, Lτ, Norbitals)
    Λ = zeros(E, Lτ, Norbitals)
    u = zeros(Complex{E}, Lτ, Norbitals)
    u′ = zeros(Complex{E}, Lτ, Norbitals)
    u″ = zeros(Complex{E}, Lτ, Norbitals)

    # initialize Λ matrix
    update_Λ!(Λ, electron_phonon_parameters)

    # initialize exact fourier accelerator
    efa = SmoQyDQMC.ExactFourierAccelerator(Ω, M, β, Δτ, η)

    return EFAPFFHMCUpdater(Nt, Δt, δ, x0, p, ∂S∂x, Λ, Φ, u, u′, u″, efa)
end


# Perform HMC update
function hmc_update!(
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAPFFHMCUpdater{E};
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_det_matrix::AbstractFermionDetMatrix{T,E},
    rng::AbstractRNG,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ,
    preconditioner = I
) where {T, E}

    (; x0, p, ∂S∂x, Φ, u, u′, u″, Λ, efa) = hmc_updater
    (; x, Δτ, dispersion_parameters, phonon_parameters) = electron_phonon_parameters

    # add a bit of noise to the time-step Δt
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)

    # record initial phonon configuration
    copyto!(x0, x)

    # initialize Λ matrix to make sure it is up to date.
    update_Λ!(Λ, electron_phonon_parameters)

    # sample Φ fields
    Sf = sample_Φ!(Φ, fermion_det_matrix, Λ, rng)

    # calculate initial bosonic action
    Sb = SmoQyDQMC.bosonic_action(electron_phonon_parameters, holstein_correction = false)

    # calculate total initial action
    S = Sf + Sb

    # initialize momentum p
    K = SmoQyDQMC.initialize_momentum!(p, efa, rng)

    # Calculate total initial energy
    H = S + K

    # evolve momentum and phonon fields according to bosonic action and update the
    # fermion path integrals to reflect the change in the phonon fields
    SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x, -1)
    SmoQyDQMC.evolve_eom!(x, p, Δt/2, efa)
    recenter!(x)
    SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x, +1)
    update!(fermion_det_matrix, fermion_path_integral)
    update_Λ!(Λ, electron_phonon_parameters)

    # average iterages initlized to zero
    iters_avg = zero(E)

    # iterate over HMC time-steps
    for t in 1:Nt

        # initialize derivative of action to zero
        fill!(∂S∂x, 0)

        # calculate derivative of fermionic action for spin-up electrons
        Sf, iters, ϵ = calculate_∂Sf∂x!(∂S∂x, Φ, Λ, fermion_det_matrix, electron_phonon_parameters, preconditioner, rng, u, u′, u″)
        iters_avg += iters / Nt

        # calculate the anharmonic phonon potential contribution to the action derivative
        SmoQyDQMC.eval_derivative_anharmonic_action!(∂S∂x, x, Δτ, phonon_parameters)

        # calculate the dispersive phonon potential contribution to the action derivative
        SmoQyDQMC.eval_derivative_dispersive_action!(∂S∂x, x, Δτ, dispersion_parameters, phonon_parameters)

        # update momentum
        @. p = p - Δt * ∂S∂x

        # evolve momentum and phonon fields according to bosonic action and update the
        # fermion path integrals to reflect the change in the phonon fields
        SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x, -1)
        Δt′ = (t==Nt) ? Δt/2 : Δt
        SmoQyDQMC.evolve_eom!(x, p, Δt′, efa)
        recenter!(x)
        SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x, +1)
        update!(fermion_det_matrix, fermion_path_integral)
        update_Λ!(Λ, electron_phonon_parameters)
    end

    # calculate final spin-up fermionic action
    Sf′, iters, ϵ = calculate_Ψ!(u, Φ, Λ, fermion_det_matrix, preconditioner, rng, power = 2.0)

    # calculate final bosonic action
    Sb′ = SmoQyDQMC.bosonic_action(electron_phonon_parameters, holstein_correction = false)

    # calculate final total action
    S′ = Sf′ + Sb′

    # calculate final total kinetic energy
    K′ = SmoQyDQMC.kinetic_energy(p, efa)

    # calculate total final energy
    H′= S′ + K′

    # calculate change in energy from initial to final state
    ΔH = H′ - H

    # calculate the acceptance probability
    P = min(1.0, exp(-ΔH))

    # determine if update accepted
    accepted = rand(rng) < P

    # if update is rejected
    if !(accepted)

        # update fermion path integrals to reflect initial phonon configuration
        SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x0, x)

        # update fermion determinant matrix to reflect initial phonon configuration
        update!(fermion_det_matrix, fermion_path_integral)

        # revert to initial phonon configuration
        copyto!(x, x0)
    end

    return accepted, iters_avg
end


# sample the pseudofermion field as Φ = Aᵀ⋅R = Λᵀ⋅Mᵀ⋅R
function sample_Φ!(
    Φ::AbstractMatrix{Complex{E}},
    fdm::AbstractFermionDetMatrix{T},
    Λ::AbstractMatrix{E},
    rng::AbstractRNG
) where {T<:Number, E<:AbstractFloat}

    # initialize R
    randn!(rng, Φ)
    # Sf = |R|²
    Sf = dot(Φ,Φ)
    # Mᵀ⋅R
    lmul_Mt!(fdm, Φ)
    # Φ = Λᵀ⋅Mᵀ⋅R
    mul_Λᵀ!(Φ, Λ, Φ)

    return real(Sf)
end


# calcualte the derivative of the fermionic action for a single spin species
function calculate_∂Sf∂x!(
    ∂Sf∂x::AbstractMatrix{E},
    Φ::AbstractMatrix{Complex{E}},
    Λ::AbstractMatrix{E},
    fdm::AbstractFermionDetMatrix{T},
    elph::ElectronPhononParameters{T,E},
    P,
    rng::AbstractRNG,
    u::AbstractMatrix{Complex{E}},
    u′::AbstractMatrix{Complex{E}},
    u″::AbstractMatrix{Complex{E}}
) where {T<:Number, E<:AbstractFloat}

    # Note: A = M⋅Λ <==> Aᵀ = Λᵀ⋅Mᵀ
    # Rename vectors for convenience
    Ψ, ΛΨ, AΨ, MᵀAΨ = u, u′, u″, u′

    # Calculate Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    Sf, iters, ϵ = calculate_Ψ!(Ψ, Φ, Λ, fdm, P, rng, power = 1.0)

    # Calculate Λ⋅Ψ
    mul_Λ!(ΛΨ, Λ, Ψ)
    # Calculate A⋅Ψ = M⋅Λ⋅Ψ
    mul_M!(AΨ, fdm, ΛΨ)
    # Calculate ∂Sf/∂x = -2⋅Re([A⋅Ψ]ᵀ⋅[∂M/∂x]⋅[Λ⋅Ψ])
    mul_νRe∂M∂x!(∂Sf∂x, -2.0, AΨ, ΛΨ, fdm, elph)
    
    # Calculate Mᵀ⋅A⋅Ψ = Mᵀ⋅M⋅Λ⋅Ψ
    mul_Mt!(MᵀAΨ, fdm, AΨ)
    # Calculate ∂Sf/∂x = -2⋅Re([A⋅Ψ]ᵀ⋅[∂M/∂x]⋅[Λ⋅Ψ]) - 2⋅Re([Mᵀ⋅A⋅Ψ]ᵀ⋅[∂Λ/∂x]⋅[Ψ]) = -2⋅Re([A⋅Ψ]ᵀ⋅[∂A/∂x]⋅[Ψ])
    mul_νRe∂Λ∂x!(∂Sf∂x, -2.0, MᵀAΨ, Ψ, Λ, elph)

    return Sf, iters, ϵ
end


# calculate Ψ vector
function calculate_Ψ!(
    Ψ::AbstractVecOrMat{Complex{E}},
    Φ::AbstractVecOrMat{Complex{E}},
    Λ::AbstractMatrix{E},
    MᵀM::AbstractFermionDetMatrix{T},
    preconditioner,
    rng::AbstractRNG;
    power::E = 1
) where {T<:Number, E<:AbstractFloat}

    tol = MᵀM.cgs.tol^power
    # Ψ = Λ⁻ᵀ⋅Φ
    ldiv_Λᵀ!(Ψ, Λ, Φ)
    # Ψ = [Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ
    # EXPENSIVE PART, AS REQUIRES CONJUGATE GRADIENT SOLVE!!!
    iters, ϵ = ldiv!(
        Ψ, MᵀM, Ψ,
        tol = tol,
        preconditioner = preconditioner,
        rng = rng,
    )
    # Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    ldiv_Λ!(Ψ, Λ, Ψ)
    # Sf = Φᵀ⋅Ψ = Φᵀ⋅[Aᵀ⋅A]⁻¹⋅Φ
    Sf = dot(Φ,Ψ)
    @assert abs(1e-6 * real(Sf)) > abs(imag(Sf)) "Complex Fermionic Action, Sf = $Sf"

    return real(Sf), iters, ϵ
end