@doc raw"""
    swap_update!(
        # ARGUMENTS
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::EFAPFFHMCUpdater{E};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_det_matrix::FermionDetMatrix{T,E},
        rng::AbstractRNG,
        preconditioner = I,
        phonon_type_pairs = nothing
    ) where {T<:Number, E<:AbstractFloat}

Randomly sample a pairs of phonon modes and exchange the phonon fields associated with the pair of phonon modes.
The argument `phonon_type_pairs` specifies pairs phonon IDs that are used to randomly samples a pairs of phonon modes.
If `phonon_type_pairs = nothing`, then all possible pairs of phonon types/IDs are allowed. This function returns a tuple
containing `(accepted, iters)`, where `accepted` is a boolean indicating whether the update was accepted or rejected, and
`iters` is the number of CG iterations performed to calculate the fermionic action.
"""
function swap_update!(
    # ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAPFFHMCUpdater{E};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    rng::AbstractRNG,
    preconditioner = I,
    phonon_type_pairs = nothing
) where {T<:Number, E<:AbstractFloat}

    (; u, Φ, Λ) = hmc_updater
    phonon_parameters = electron_phonon_parameters.phonon_parameters
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    x = electron_phonon_parameters.x

    # get the mass associated with each phonon
    M = phonon_parameters.M

    # get the number of phonon modes per unit cell
    nphonon = phonon_parameters.nphonon

    # total number of phonon modes
    Nphonon = phonon_parameters.Nphonon

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # sample random phonon mode
    phonon_mode_i, phonon_mode_j = SmoQyDQMC._sample_phonon_mode_pair(rng, nphonon, Nunitcells, M, phonon_type_pairs)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (phonon_mode_i in holstein_parameters.coupling_to_phonon) || (phonon_mode_j in holstein_parameters.coupling_to_phonon)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (phonon_mode_i in ssh_parameters.coupling_to_phonon) || (phonon_mode_j in ssh_parameters.coupling_to_phonon)

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode_i, :]
    x_j = @view x[phonon_mode_j, :]

    # initialize Λ matrix to make sure it is up to date.
    update_Λ!(Λ, electron_phonon_parameters)

    # sample Φ fields
    Sf = sample_Φ!(Φ, fermion_det_matrix, Λ, rng)

    # calculate the initial bosonic action
    Sb = SmoQyDQMC.bosonic_action(electron_phonon_parameters, holstein_correction = false)

    # Calculate total initial action
    S = Sf + Sb

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # swap phonon fields
    SmoQyDQMC.swap!(x_i, x_j)

    # update the fermion path integrals to reflect new phonon field configuration
    if calculate_exp_V
        SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, +1)
    end
    if calculate_exp_K
        SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, +1)
    end

    # update the fermion determinant matrix
    update!(fermion_det_matrix, fermion_path_integral)

    # initialize Λ matrix to make sure it is up to date.
    update_Λ!(Λ, electron_phonon_parameters)

    # calculate final spin-up fermionic action
    Sf′, iters, ϵ = calculate_Ψ!(u, Φ, Λ, fermion_det_matrix, preconditioner, rng, power = 2.0)

    # calculate the initial bosonic action
    Sb′ = SmoQyDQMC.bosonic_action(electron_phonon_parameters, holstein_correction = false)

    # calculate total final action
    S′ = Sf′ + Sb′

    # calculate the change in action
    ΔS = S′ - S

    # calculate acceptance probability
    P = min(1.0, exp(-ΔS))

    # if update is accepted
    if rand(rng) < P
        accepted = true
    # if update is rejected
    else
        accepted = false
        # substract off the effect of the current phonon configuration on the fermion path integrals
        if calculate_exp_V
            SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, -1)
        end
        if calculate_exp_K
            SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, -1)
        end
        # swap phonon fields
        SmoQyDQMC.swap!(x_i, x_j)
        # update the fermion path integrals to reflect new phonon field configuration
        if calculate_exp_V
            SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, +1)
        end
        if calculate_exp_K
            SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, +1)
        end
        # update the fermion determinant matrix
        update!(fermion_det_matrix, fermion_path_integral)
    end

    return accepted, iters
end