@doc raw"""
    update_chemical_potential!(
        fermion_det_matrix::FermionDetMatrix{T,E},
        greens_estimator::GreensEstimator{E,D};
        chemical_potential_tuner::MuTunerLogger{E,T},
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{T,E},
        preconditioner = I,
        rng::AbstractRNG = Random.default_rng(),
        update_greens_estimator::Bool = true
    ) where {D, T<:Number, E<:AbstractFloat}

Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
If `update_greens_estimator = true`, then `greens_estimator` is initialized to reflect the current
state of the `fermion_det_matrix`.
"""
function update_chemical_potential!(
    fermion_det_matrix::FermionDetMatrix{T,E},
    greens_estimator::GreensEstimator{E,D};
    chemical_potential_tuner::MuTunerLogger{E,T},
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    preconditioner = I,
    rng::AbstractRNG = Random.default_rng(),
    update_greens_estimator::Bool = true
) where {D, T<:Number, E<:AbstractFloat}

    # initialize the Green's function estimator to reflect the current fermion determinant matrix
    if update_greens_estimator
        update_greens_estimator!(
            greens_estimator, fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng
        )
    end

    # record the initial chemical potential
    μ′ = tight_binding_parameters.μ

    # calculate sign
    sgn = one(E)

    # calculate average density
    n = 2 * measure_n(greens_estimator)

    # calculate ⟨N²⟩
    Nsqrd = measure_Nsqrd(greens_estimator)

    # update the chemical potential
    μ = MuTuner.update!(chemical_potential_tuner, n, Nsqrd, sgn)

    # update tight binding parameter chemical potential
    tight_binding_parameters.μ = μ

    # update fermion path integral
    V = fermion_path_integral.V
    @. V += -μ + μ′

    # update the fermion determinant matrix
    update!(fermion_det_matrix, fermion_path_integral)

    return nothing
end