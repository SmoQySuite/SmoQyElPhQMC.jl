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

    # make euqal-time and time-displaced measurements
    make_correlation_measurements!(
        measurement_container,
        greens_estimator,
        model_geometry,
        tight_binding_parameters,
        fermion_path_integral
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
    global_measurements["action_bosonic"] += SmoQyDQMC.bosonic_action(electron_phonon_parameters)
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

    # make density related local measurements
    for n in 1:norbital
        # measure density
        density = measure_n(greens_estimator, n)
        local_measurements["density_up"][n] += density
        local_measurements["density_dn"][n] += density
        local_measurements["density"][n] += 2* density
        # measure double occupancy
        local_measurements["double_occ"][n] += measure_double_occ(greens_estimator, n)
    end

    # make tight-binding measurements
    make_tight_binding_measurements!(
        local_measurements,
        model_geometry,
        tight_binding_parameters,
        fermion_path_integral,
        greens_estimator,
    )

    # make electron-phonon measurements
    make_electron_phonon_measurements!(
        local_measurements,
        model_geometry,
        electron_phonon_parameters,
        greens_estimator,
    )

    return nothing
end

# make correlation measurements
function make_correlation_measurements!(
    measurement_container::NamedTuple,
    greens_estimator::GreensEstimator{E},
    model_geometry::ModelGeometry{D,E},
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E}
) where {D, E<:AbstractFloat, T<:Number}

    # get defined bonds on model geometry
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # get dimensions of lattice
    L = model_geometry.lattice.L

    # get length of imaginary time axis
    Lτ = fermion_path_integral.Lτ

    # get correlation containers
    (; time_displaced_correlations, equaltime_correlations) = measurement_container

    # get temporarary correlation container
    tmp = greens_estimator.tmp

    # get all correlation measurements
    equaltime_measurements = keys(equaltime_correlations)
    time_displaced_measurements = keys(time_displaced_correlations)
    measurements = union(equaltime_measurements, time_displaced_measurements)

    # iterate over all correlation function measurements
    for correlation in measurements

        # get the relevant id pairs for current correlation measurement
        id_pairs = get_id_pairs(equaltime_correlations, time_displaced_correlations, correlation)

        if (correlation == "greens") || (correlation == "greens_up") || (correlation == "greens_dn")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                id_pair = id_pairs[i]
                measure_GΔ0!(tmp, greens_estimator, id_pair)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end 

        elseif (correlation == "density_upup") || (correlation == "density_dndn")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                b, a = id_pairs[i]
                measure_density_correlation!(tmp, greens_estimator, a, b, +1, +1)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "density_updn") || (correlation == "density_dnup")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                b, a = id_pairs[i]
                measure_density_correlation!(tmp, greens_estimator, a, b, +1, -1)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                b, a = id_pairs[i]
                measure_density_correlation!(tmp, greens_estimator, a, b)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                id_pair = id_pairs[i]
                b″ = bonds[id_pair[1]]
                b′ = bonds[id_pair[2]]
                measure_pair_correlation!(tmp, greens_estimator, b′, b″)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "spin_z") || (correlation == "spin_x")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                b, a = id_pairs[i]
                measure_spin_correlation!(tmp, greens_estimator, a, b)
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "bond_upup") || (correlation == "bond_dndn")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                id_pair = id_pairs[i]
                b″ = bonds[id_pair[1]]
                b′ = bonds[id_pair[2]]
                measure_bond_correlation!(
                    tmp, greens_estimator,
                    b′, b″, +1, +1, 1.0
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "bond_updn") || (correlation == "bond_dnup")

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                id_pair = id_pairs[i]
                b″ = bonds[id_pair[1]]
                b′ = bonds[id_pair[2]]
                measure_bond_correlation!(
                    tmp, greens_estimator,
                    b′, b″, +1, -1, 1.0
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                id_pair = id_pairs[i]
                b″ = bonds[id_pair[1]]
                b′ = bonds[id_pair[2]]
                measure_bond_correlation!(
                    tmp, greens_estimator,
                    b′, b″
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "current_upup") || (correlation == "current_dndn")

            (; bond_ids, bond_slices) = tight_binding_parameters
            t = fermion_path_integral.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                b″ = bonds[bond_id_0]
                b′ = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                t0 = reshape(view(t, bond_slices[hopping_id_0], :), (L...,Lτ))
                t1 = reshape(view(t, bond_slices[hopping_id_1], :), (L...,Lτ))
                # reshape hopping into approxpropriate shape
                t″ = PermutedDimsArray(t0, (D+1, 1:D...))
                t′ = PermutedDimsArray(t1, (D+1, 1:D...))
                # measure current correlation
                measure_current_correlation!(
                    tmp, greens_estimator,
                    b′, b″, t′, t″, +1, +1
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif (correlation == "current_updn") || (correlation == "current_dnup")

            (; bond_ids, bond_slices) = tight_binding_parameters
            t = fermion_path_integral.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                b″ = bonds[bond_id_0]
                b′ = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                t0 = reshape(view(t, bond_slices[hopping_id_0], :), (L...,Lτ))
                t1 = reshape(view(t, bond_slices[hopping_id_1], :), (L...,Lτ))
                # reshape hopping into approxpropriate shape
                t″ = PermutedDimsArray(t0, (D+1, 1:D...))
                t′ = PermutedDimsArray(t1, (D+1, 1:D...))
                # measure current correlation
                measure_current_correlation!(
                    tmp, greens_estimator,
                    b′, b″, t′, t″, +1, -1
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters
            t = fermion_path_integral.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                b″ = bonds[bond_id_0]
                b′ = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                t0 = reshape(view(t, bond_slices[hopping_id_0], :), (L...,Lτ))
                t1 = reshape(view(t, bond_slices[hopping_id_1], :), (L...,Lτ))
                # reshape hopping into approxpropriate shape
                t″ = PermutedDimsArray(t0, (D+1, 1:D...))
                t′ = PermutedDimsArray(t1, (D+1, 1:D...))
                # measure current correlation
                measure_current_correlation!(
                    tmp, greens_estimator,
                    b′, b″, t′, t″
                )
                copyto_correlation_container!(equaltime_correlations, time_displaced_correlations, correlation, tmp, i)
            end
        end
    end

    return nothing
end


# get the id pairs associated with correlation measurements
function get_id_pairs(
    equaltime_correlations::Dict{String, CorrelationContainer{D,T}},
    time_displaced_correlations::Dict{String, CorrelationContainer{Dp1,T}},
    measurement::String
) where {D, Dp1, T}

    if measurement in keys(equaltime_correlations)
        id_pairs = equaltime_correlations[measurement].id_pairs
    elseif measurement in keys(time_displaced_correlations)
        id_pairs = time_displaced_correlations[measurement].id_pairs
    end

    return id_pairs
end


# record the correlation measurement stored in tmp to appropriate correlation containers
function copyto_correlation_container!(
    equaltime_correlations::Dict{String, CorrelationContainer{D,T}},
    time_displaced_correlations::Dict{String, CorrelationContainer{Dp1,T}},
    measurement::String,
    tmp::AbstractArray{Complex{T}, Dp1},
    i::Int
) where {D, Dp1, T<:AbstractFloat}

    # record equal-time correlation measurements
    if measurement in keys(equaltime_correlations)
        correlations = equaltime_correlations[measurement].correlations[i]
        eqltm_tmp = selectdim(tmp, Dp1, 1)
        @. correlations += eqltm_tmp
    end

    # record time-displaced correlation measurements
    if measurement in keys(time_displaced_correlations)
        correlations = time_displaced_correlations[measurement].correlations[i]
        @. correlations += tmp
    end

    return nothing
end