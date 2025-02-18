@doc raw"""
Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetG, sgndetG, δG, δθ)`.
"""
function make_measurements!(
    measurement_container::NamedTuple,
    fermion_det_matrix::FermionDetMatrix{T,E},
    greens_estimator::GreensEstimator{E, D};
    # Keyword Arguments Start Here
    model_geometry::ModelGeometry{D,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    tight_binding_parameters::TightBindingParameters{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    preconditioner = I,
    rng::AbstractRNG = Random.default_rng()
) where {T<:Number, E<:AbstractFloat, D}

    # initialize the Green's function estimator to reflect the current
    # fermion determinant matrix
    update_greens_estimator!(
        greens_estimator, fermion_det_matrix,
        preconditioner = preconditioner,
        rng = rng
    )

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(
        global_measurements,
        tight_binding_parameters,
        electron_phonon_parameters,
        greens_estimator,
    )

    # make local measurements
    local_measurements = measurement_container.local_measurements
    make_local_measurements!(
        local_measurements,
        model_geometry,
        tight_binding_parameters,
        electron_phonon_parameters,
        fermion_path_integral,
        greens_estimator,
    )

    return nothing
end

# make global measurements
function make_global_measurements!(
    global_measurements::Dict{String, Complex{E}},
    tight_binding_parameters::TightBindingParameters{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    greens_estimator::GreensEstimator{E},
) where {E<:AbstractFloat, T<:Number}

    global_measurements["sgn"] += 1.0
    global_measurements["sgndetGup"] = NaN
    global_measurements["sgndetGdn"] = NaN
    global_measurements["logdetGup"] = NaN
    global_measurements["logdetGdn"] = NaN
    global_measurements["action_fermionic"] = NaN
    global_measurements["action_bosonic"] += bosonic_action(electron_phonon_parameters)
    global_measurements["action_total"] = NaN
    density = measure_n(greens_estimator)
    global_measurements["density_up"] += density
    global_measurements["density_dn"] += density
    global_measurements["density"] += 2 * density
    global_measurements["double_occ"] += measure_double_occ(greens_estimator)
    global_measurements["Nsqrd"] += measure_Nsqrd(greens_estimator)
    global_measurements["chemical_potential"] += tight_binding_parameters.μ

    return nothing
end

# make local measurements
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::TightBindingParameters{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    greens_estimator::GreensEstimator{E},
) where {E<:AbstractFloat, T<:Number, D, N}

    # number of orbitals per unit cell
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    norbital = unit_cell.n

    # STANDARD LOCAL MEASUREMENTS

    # iterate over orbital species
    for n in 1:norbital
        # measure density
        density = measure_n(greens_estimator, n)
        local_measurements["density_up"][n] += density
        local_measurements["density_dn"][n] += density
        local_measurements["density"][n] += 2* density
        # measure double occupancy
        local_measurements["double_occ"][n] += measure_double_occ(greens_estimator, n)
    end

    # TIGHT-BINDING LOCAL MEASUREMENTS

    # measure on-site energy
    for n in 1:norbital
        ϵ_onsite = measure_onsite_energy(greens_estimator, tight_binding_parameters, n)
        local_measurements["onsite_energy_up"][n] += ϵ_onsite
        local_measurements["onsite_energy_dn"][n] += ϵ_onsite
        local_measurements["onsite_energy"][n] += 2*ϵ_onsite
    end

    # number of types of hopping
    bond_ids = tight_binding_parameters_up.bond_ids
    nhopping = length(tight_binding_parameters_up.bond_ids)

    # measure hopping energy
    if nhopping > 0
        for n in 1:nhopping
            # measure bare hopping amplitude
            ϵ_bare_hopping = measure_bare_hopping_energy(greens_estimator, tight_binding_parameters, model_geometry, n)
            local_measurements["bare_hopping_energy_up"][n] += ϵ_bare_hopping
            local_measurements["bare_hopping_energy_dn"][n] += ϵ_bare_hopping
            local_measurements["bare_hopping_energy"][n] += 2*ϵ_bare_hopping
            # measure hopping amplitude
            ϵ_hopping = measure_hopping_energy(greens_estimator, fermion_path_integral, tight_binding_parameters, model_geometry, n)
            local_measurements["bare_hopping_energy_up"][n] += ϵ_hopping
            local_measurements["bare_hopping_energy_dn"][n] += ϵ_hopping
            local_measurements["bare_hopping_energy"][n] += 2*ϵ_hopping
        end

    end

    return nothing
end