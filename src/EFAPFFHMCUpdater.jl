@doc raw"""
    EFAPFFHMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

A type for performing exact fourier acceleration pseudofermion field hybrid/Hamiltonian
Monte Carlo (EFA-PFF-HMC) updates to the phonon fields. 
"""
struct EFAPFFHMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

    Nt::Int
    Δt::E
    δ::E
    x0::Matrix{E}
    p::Matrix{E}
    ∂S∂x::Matrix{E}
    Φ₊::Matrix{T}
    Φ₋::Matrix{T}
    Λ::Matrix{T}
    R::Matrix{E}
    u::Matrix{T}
    u′::Matrix{T}
    u″::Matrix{T}
    efa::SmoQyDQMC.ExactFourierAccelerator{E, PFFT, PIFFT}
    cgs::ConjugateGradientSolver{T,E}
end

@doc raw"""
    EFAPFFHMCUpdater(;
        # KEYWORD ARGUMENTS
        electron_phonon_parameters::ElectronPhononParameters{T},
        fermion_det_matrix::AbstractFermionDetMatrix{T},
        Nt::Int,
        Δt::E,
        η::E,
        δ::E = 0.05,
        tol::E = 1e-6,
        maxiter::Int = size(fermion_det_matrix,1)
    ) where {T<:Number, E<:AbstractFloat}

Initialize an instance of the [`EFAPFFHMCUpdater`](@ref) type for performing
exact fourier acceleration pseudofermion field hybrid/Hamiltonian Monte Carlo (EFA-PFF-HMC)
updates to the phonon fields.

# Keyword Arguments

- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Contains electron-phonon model parameters, including phonon field configuration.
- `fermion_det_matrix::AbstractFermionDetMatrix{T,E}`: A type representing the Fermion determinant matrix.
- `Nt::Int`: Number of timesteps to perform in EFA-PFF-HMC update.
- `Δt::E`: Size of timestep used in EFA-PFF-HMC update.
- `η::E = 1.0`: Regularization parameter used in Exact Fourer Accerleration (EFA).
- `δ::E = 0.05`: Amount by which the timestep `Δt` is randomized before each update.
- `tol::E = 1e-6`: Tolerance used when performing conjugate gradient solves to calcualte the fermionic action.
- `maxiter::Int = size(fermion_det_matrix,1)`: Max number of iterations when performing conjugate gradient solves to calcualte the fermionic action.
"""
function EFAPFFHMCUpdater(;
    # KEYWORD ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T},
    fermion_det_matrix::AbstractFermionDetMatrix{T},
    Nt::Int,
    Δt::E,
    η::E = 0.0,
    δ::E = 0.05,
    tol::E = 1e-6,
    maxiter::Int = size(fermion_det_matrix,1)
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
    Φ₊ = zeros(T, Lτ, Norbitals)
    Φ₋ = zeros(T, Lτ, Norbitals)
    Λ = zeros(T, Lτ, Norbitals)
    R = zeros(E, Lτ, Norbitals)
    u = zeros(T, Lτ, Norbitals)
    u′ = zeros(T, Lτ, Norbitals)
    u″ = zeros(T, Lτ, Norbitals)

    # initialize Λ matrix
    update_Λ!(Λ, electron_phonon_parameters)

    # initialize exact fourier accelerator
    efa = SmoQyDQMC.ExactFourierAccelerator(Ω, M, β, Δτ, η)

    # initialize conjugate gradient solver
    cgs = ConjugateGradientSolver(Φ₊, maxiter = maxiter, tol = tol)

    return EFAPFFHMCUpdater(Nt, Δt, δ, x0, p, ∂S∂x, Φ₊, Φ₋, Λ, R, u, u′, u″, efa, cgs)
end

@doc raw"""
    hmc_update!(;
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::EFAPFFHMCUpdater{T,E};
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_det_matrix::AbstractFermionDetMatrix{T},
        rng::AbstractRNG,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ,
        preconditioner = I
    ) where {T, E}

Perform exact fourier acceleration pseudofermion field hybrid/Hamiltonian Monte Carlo (EFA-PFF-HMC) update to the phonon fields.
"""
function hmc_update!(
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAPFFHMCUpdater{T,E};
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_det_matrix::AbstractFermionDetMatrix{T},
    rng::AbstractRNG,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ,
    preconditioner = I
) where {T, E}

    (; x0, p, ∂S∂x, Φ₊, Φ₋, R, u, u′, u″, Λ, efa, cgs) = hmc_updater
    (; x, Δτ, dispersion_parameters, phonon_parameters) = electron_phonon_parameters

    # add a bit of noise to the time-step Δt
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)

    # record initial phonon configuration
    copyto!(x0, x)

    # initialize Λ matrix to make sure it is up to date.
    update_Λ!(Λ, electron_phonon_parameters)

    # intialize preconditioner
    update_preconditioner!(preconditioner, fermion_det_matrix, rng)

    # sample Φ fields
    Sup = sample_Φ!(Φ₊, fermion_det_matrix, Λ, R, rng)
    Sdn = sample_Φ!(Φ₋, fermion_det_matrix, Λ, R, rng)

    # calculate initial fermionic action
    Sf = Sup + Sdn

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
    update_preconditioner!(preconditioner, fermion_det_matrix, rng)
    update_Λ!(Λ, electron_phonon_parameters)

    # iterate over HMC time-steps
    for t in 1:Nt

        # initialize derivative of action to zero
        fill!(∂S∂x, 0)

        # calculate derivative of fermionic action for spin-up electrons
        calculate_∂Sf∂x!(∂S∂x, Φ₊, Λ, fermion_det_matrix, electron_phonon_parameters, cgs, preconditioner, u, u′, u″)

        # calculate derivative of fermionic action for spin-down electrons
        calculate_∂Sf∂x!(∂S∂x, Φ₋, Λ, fermion_det_matrix, electron_phonon_parameters, cgs, preconditioner, u, u′, u″)

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
        update_preconditioner!(preconditioner, fermion_det_matrix, rng)
        update_Λ!(Λ, electron_phonon_parameters)
    end

    # calculate final spin-up fermionic action
    Sup′ = calculate_Ψ!(u, Φ₊, Λ, fermion_det_matrix, cgs, preconditioner)

    # calculate final spin-down fermionic action
    Sdn′ = calculate_Ψ!(u, Φ₋, Λ, fermion_det_matrix, cgs, preconditioner)

    # calculate final fermionic action
    Sf′ = Sup′ + Sdn′

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

        # update preconditioner to reflect initial phonon configuration
        update_preconditioner!(preconditioner, fermion_det_matrix, rng)

        # revert to initial phonon configuration
        copyto!(x, x0)
    end

    return accepted
end

# sample the pseudofermion field as Φ = Aᵀ⋅R = Λᵀ⋅Mᵀ⋅R
function sample_Φ!(
    Φ::AbstractMatrix{T},
    fdm::AbstractFermionDetMatrix{T},
    Λ::AbstractMatrix{T},
    R::AbstractMatrix{E},
    rng::AbstractRNG
) where {T<:Number, E<:AbstractFloat}

    # initialize R₊
    randn!(rng, R)
    # S = |R|²
    S = dot(R,R)/2
    # Mᵀ⋅R
    mul_Mt!(Φ, fdm, R)
    # Φ = Λᵀ⋅Mᵀ⋅R
    mul_Λᵀ!(Φ, Λ, Φ)

    return S
end

# calcualte the derivative of the fermionic action for a single spin species
function calculate_∂Sf∂x!(
    ∂Sf∂x::AbstractMatrix{E},
    Φ::AbstractMatrix{T},
    Λ::AbstractMatrix{T},
    fdm::AbstractFermionDetMatrix{T},
    elph::ElectronPhononParameters{T,E},
    cgs::ConjugateGradientSolver{T,E},
    P,
    u::AbstractMatrix{T},
    u′::AbstractMatrix{T},
    u″::AbstractMatrix{T}
) where {T<:Number, E<:AbstractFloat}

    # Note: A = M⋅Λ <==> Aᵀ = Λᵀ⋅Mᵀ
    # Rename vectors for convenience
    Ψ, ΛΨ, AΨ, MᵀAΨ = u, u′, u″, u′

    # Calculate Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    Sf = calculate_Ψ!(Ψ, Φ, Λ, fdm, cgs, P)

    # Calculate Λ⋅Ψ
    mul_Λ!(ΛΨ, Λ, Ψ)
    # Calculate A⋅Ψ = M⋅Λ⋅Ψ
    mul_M!(AΨ, fdm, ΛΨ)
    # Calculate ∂Sf/∂x = [A⋅Ψ]ᵀ⋅[-∂M/∂x]⋅[Λ⋅Ψ]
    mul_n∂M∂x!(∂Sf∂x, AΨ, ΛΨ, fdm, elph)
    
    # Calculate Mᵀ⋅A⋅Ψ = Mᵀ⋅M⋅Λ⋅Ψ
    mul_Mt!(MᵀAΨ, fdm, AΨ)
    # Calculate ∂Sf/∂x = [A⋅Ψ]ᵀ⋅[-∂M/∂x]⋅[Λ⋅Ψ] + [Mᵀ⋅A⋅Ψ]ᵀ⋅[-∂Λ/∂x]⋅[Ψ] = -[A⋅Ψ]ᵀ⋅[∂A/∂x]⋅[Ψ]
    mul_n∂Λ∂x!(∂Sf∂x, MᵀAΨ, Ψ, Λ, elph)

    return Sf
end

# calculate Ψ vector
function calculate_Ψ!(
    Ψ::AbstractVecOrMat{T},
    Φ::AbstractVecOrMat{T},
    Λ::AbstractMatrix{T},
    MᵀM::AbstractFermionDetMatrix{T},
    cgs::ConjugateGradientSolver{T,E},
    P # Preconditioner
) where {T<:Number, E<:AbstractFloat}

    # Ψ = Λ⁻ᵀ⋅Φ
    ldiv_Λᵀ!(Ψ, Λ, Φ)
    # Ψ = [Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ
    # THE EXPENSIVE PART, A CONJUGATE GRADIENT SOLVE!!!
    iters, ϵ = cg_solve!(Ψ, MᵀM, Ψ, cgs, P)
    # Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    ldiv_Λ!(Ψ, Λ, Ψ)
    # Sf = (Φᵀ⋅Ψ)/2 = (Φᵀ⋅[Aᵀ⋅A]⁻¹⋅Φ)/2
    Sf = dot(Φ,Ψ)/2

    return Sf
end

# default update! preconditioner method does nothing
update_preconditioner!(P, fdm, rng) = nothing