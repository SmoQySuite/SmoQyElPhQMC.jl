@doc raw"""
    struct EFAPFFHMCUpdater{T<:AbstractFloat, PFFT, PIFFT}

Type to define how to perform an Hybrid/Hamiltonian Monte Carlo (HMC) updates of the
phonon fields using a fermionic action formulated by introducing a complex pseudofermion
field (PFF) `\Phi`. Exact Fourier Acceleration (EFA) is also used to more efficiently sample the
phonon fields.
"""
struct EFAPFFHMCUpdater{T<:AbstractFloat, PFFT, PIFFT}

    Nt::Int
    Δt::T
    δ::T
    x0::Matrix{T}
    p::Matrix{T}
    ∂S∂x::Matrix{T}
    efa::SmoQyDQMC.ExactFourierAccelerator{T, PFFT, PIFFT}
end


@doc raw"""
    EFAPFFHMCUpdater(;
        # Keyword Arguments
        electron_phonon_parameters::ElectronPhononParameters{T},
        Nt::Int,
        Δt::E,
        η::E = 0.0,
        δ::E = 0.05
    ) where {T<:Number, E<:AbstractFloat}

Initialize an instance of [`EFAPFFHMCUpdater`](@ref) type, defining an EFA-PFF-HMC update.

# Keyword Arguments
- `electron_phonon_parameters::ElectronPhononParameters{T}`: Parameters defining the electron-phonon model.
- `Nt::Int`: Number of HMC time-steps.
- `Δt::E`: Time-step for HMC update.
- `η::E = 0.0`: Regularization parameter for EFA.
- `δ::E = 0.05`: Fractional noise to add to the time-step `Δt`.
"""
function EFAPFFHMCUpdater(;
    # Keyword Arguments
    electron_phonon_parameters::ElectronPhononParameters{T},
    Nt::Int,
    Δt::E,
    η::E = 0.0,
    δ::E = 0.05
) where {T<:Number, E<:AbstractFloat}

    (; β, Δτ, phonon_parameters, x) = electron_phonon_parameters
    (; Ω, M) = phonon_parameters

    # number of phonon modes and imaginary-time slices
    Nph, Lτ = size(x)

    # allocate arrays
    x0 = zeros(E, Nph, Lτ)
    p = zeros(E, Nph, Lτ)
    ∂S∂x = zeros(E, Nph, Lτ)

    # initialize exact fourier accelerator
    efa = SmoQyDQMC.ExactFourierAccelerator(Ω, M, β, Δτ, η)

    return EFAPFFHMCUpdater(Nt, Δt, δ, x0, p, ∂S∂x, efa)
end


@doc raw"""
    hmc_update!(
        # ARGUMENTS
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::EFAPFFHMCUpdater{E};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_det_matrix::FermionDetMatrix{T,E},
        pff_calculator::PFFCalculator{E},
        rng::AbstractRNG,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ,
        tol_action::E = fermion_det_matrix.cg.tol,
        tol_force::E = sqrt(fermion_det_matrix.cg.tol),
        max_iter::Int = fermion_det_matrix.cg.maxiter,
        preconditioner = I
    ) where {T, E}

Perform an EFA-PFF-HMC update to the phonon fields.
Acronym EFA-PFF-HMC stands for pseudofermion field (PPF) Hamiltonian/hyrbid Monte Carlo (HMC) update
with exact Fourier acceleration (EFA) used to reduce autocorrelation times.

# Keyword Arguments with Default Values

- `recenter!::Function = identity`: Function to recenter the phonon fields after the update.
- `Nt::Int = hmc_updater.Nt`: Number of HMC time-steps.
- `Δt::E = hmc_updater.Δt`: Time-step for HMC update.
- `δ::E = hmc_updater.δ`: Fractional noise to add to the time-step `Δt`.
- `tol_action::E = fermion_det_matrix.cg.tol`: Tolerance used in CG solve to evaluate fermionic action.
- `tol_force::E = sqrt(fermion_det_matrix.cg.tol)`: Tolerance used in CG solve to evaluate derivative of fermionic action.
- `max_iter::Int = fermion_det_matrix.cg.maxiter`: Maximum number of iterations for CG solve.
- `preconditioner = I`: Preconditioner used in CG solves.
"""
function hmc_update!(
    # ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAPFFHMCUpdater{E};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    pff_calculator::PFFCalculator{E},
    rng::AbstractRNG,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ,
    tol_action::E = fermion_det_matrix.cg.tol,
    tol_force::E = sqrt(fermion_det_matrix.cg.tol),
    maxiter::Int = fermion_det_matrix.cg.maxiter,
    preconditioner = I
) where {T, E}

    (; x0, p, ∂S∂x, efa) = hmc_updater
    (; x, Δτ, dispersion_parameters, phonon_parameters) = electron_phonon_parameters

    # add a bit of noise to the time-step Δt
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)

    # record initial phonon configuration
    copyto!(x0, x)

    # sample pseudofermion fields and get initial fermionic action
    Sf = sample_pseudofermion_fields!(
        pff_calculator, electron_phonon_parameters, fermion_det_matrix, rng
    )

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

    # average iterages initlized to zero
    iters_avg = zero(E)

    # iterate over HMC time-steps
    for t in 1:Nt

        # initialize derivative of action to zero
        fill!(∂S∂x, 0)

        # calculate derivative of fermionic action for spin-up electrons
        Sf, iters, ϵ = calculate_derivative_fermionic_action!(
            ∂S∂x, pff_calculator, electron_phonon_parameters, fermion_det_matrix, preconditioner, rng, tol_force, maxiter
        )
        iters_avg += iters / (Nt+1)

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
    end

    # calculate final fermionic action
    Sf′, iters, ϵ = calculate_fermionic_action!(
        pff_calculator, electron_phonon_parameters, fermion_det_matrix,
        preconditioner, rng, tol_action, maxiter
    )
    iters_avg += iters / (Nt+1)

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