@doc raw"""
    reflection_update!(
        # ARGUMENTS
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        pff_calculator::PFFCalculator{E};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_det_matrix::FermionDetMatrix{T,E},
        rng::AbstractRNG,
        preconditioner = I,
        tol::E = fermion_det_matrix.cg.tol,
        maxiter::Int = fermion_det_matrix.cg.maxiter,
        phonon_types = nothing
    ) where {T<:Number, E<:AbstractFloat}

Randomly sample a phonon mode in the lattice, and propose an update that reflects all the phonon fields
associated with that phonon mode ``x \rightarrow -x.`` The argument `phonon_types` specifies the phonon ID's
that are included for randomly sampling a phonon mode in the lattice to perform a swap update on. If
`phonon_types = nothing`, then all types of phonon modes are included. This function returns a tuple containing
`(accepted, iters)`, where `accepted` is a boolean indicating whether the update was accepted or rejected, and
`iters` is the number of CG iterations performed to calculate the fermionic action.
"""
function reflection_update!(
    # ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    pff_calculator::PFFCalculator{E};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    rng::AbstractRNG,
    preconditioner = I,
    tol::E = fermion_det_matrix.cg.tol,
    maxiter::Int = fermion_det_matrix.cg.maxiter,
    phonon_types = nothing
) where {T<:Number, E<:AbstractFloat}

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
    phonon_mode = SmoQyDQMC._sample_phonon_mode(rng, nphonon, Nunitcells, M, phonon_types)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (phonon_mode in holstein_parameters.coupling_to_phonon)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (phonon_mode in ssh_parameters.coupling_to_phonon)

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode, :]

    # sample pseudofermion fields and calculate initial fermionic action
    Sf = sample_pseudofermion_fields!(
        pff_calculator, electron_phonon_parameters, fermion_det_matrix, rng
    )

    # calculate the initial bosonic action
    Sb = SmoQyDQMC.bosonic_action(electron_phonon_parameters, holstein_correction = false)

    # Calculate total initial action
    S = Sf + Sb

    # subtract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # reflection phonon fields for chosen mode
    @. x_i = -x_i

    # update the fermion path integrals to reflect new phonon field configuration
    if calculate_exp_V
        SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, +1)
    end
    if calculate_exp_K
        SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, +1)
    end

    # update the fermion determinant matrix
    update!(fermion_det_matrix, fermion_path_integral)

    # calculate final fermionic action
    Sf′, iters, ϵ = calculate_fermionic_action!(
        pff_calculator, electron_phonon_parameters, fermion_det_matrix,
        preconditioner, rng, tol, maxiter
    )

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
        # subtract off the effect of the current phonon configuration on the fermion path integrals
        if calculate_exp_V
            SmoQyDQMC.update!(fermion_path_integral, holstein_parameters, x, -1)
        end
        if calculate_exp_K
            SmoQyDQMC.update!(fermion_path_integral, ssh_parameters, x, -1)
        end
        # revert to the original phonon configuration
        @. x_i = -x_i
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