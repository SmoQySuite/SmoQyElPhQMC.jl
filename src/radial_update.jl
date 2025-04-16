@doc raw"""
    radial_update!(
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
        phonon_id::Union{Nothing,Int} = nothing,
        σ::E = 1.0
    ) where {T<:Number, E<:AbstractFloat}

Perform a radial update to the phonon fields, as described by Algorithm 1 in the paper
[arXiv:2411.18218](https://arxiv.org/abs/2411.18218).
Specifically, the proposed update to the phonon fields ``x`` is a rescaling such that
``x \rightarrow e^{\gamma} x`` where ``\gamma \sim N(0, \sigma/\sqrt{d})`` and ``d`` is
the number of phonon fields being updated.
"""
function radial_update!(
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
    phonon_id::Union{Nothing,Int} = nothing,
    σ::E = 1.0
) where {T<:Number, E<:AbstractFloat}

    phonon_parameters = electron_phonon_parameters.phonon_parameters
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    x = electron_phonon_parameters.x
    M = phonon_parameters.M

    # get the mass associated with each phonon
    M = phonon_parameters.M

    # get the number of phonon modes per unit cell
    nphonon = phonon_parameters.nphonon

    # total number of phonon modes
    Nphonon = phonon_parameters.Nphonon

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (holsein_parameters.nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (ssh_parameters.nssh > 0)

    # get phonon fields and mass for specified phonon mode if necessary
    if !isnothing(phonon_id)
        M′ = @view M[(phonon_id-1)*Nunitcells+1:phonon_id*Nunitcells]
        x′ = @view x[(phonon_id-1)*Nunitcells+1:phonon_id*Nunitcells, :]
    else
        M′ = M
        x′ = x
    end

    # number of fields to update, excluding phonon fields that correspond
    # to phonon modes with infinite mass
    d = count(m -> isfinite(m), M′)

    # calculate standard deviation for normal distribution
    σR = σ / sqrt(d)

    # randomly sample expansion/contraction coefficient
    γ = randn(rng) * σR
    expγ = exp(γ)

    # sample pseudofermion fields and calculate initial fermionic action
    Sf = sample_pseudofermion_fields!(
        pff_calculator, electron_phonon_parameters, fermion_det_matrix, rng
    )

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

    # apply expansion/contraction to phonon fields
    @. x′ = expγ * x′

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
    P = min(1.0, exp(-ΔS + d*γ))

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
        # revert to the original phonon configuration
        @. x′ = x′ / expγ
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