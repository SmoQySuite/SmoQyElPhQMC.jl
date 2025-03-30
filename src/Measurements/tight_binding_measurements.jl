# make tight-binding measurements
function make_tight_binding_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    greens_estimator::GreensEstimator{E},
) where {E<:AbstractFloat, T<:Number, D, N}

    # number of orbitals in unit cell
    norbital = greens_estimator.n

    # measure on-site energy
    for orbital_id in 1:norbital
        ϵ_onsite = measure_onsite_energy(greens_estimator, tight_binding_parameters, orbital_id)
        local_measurements["onsite_energy_up"][orbital_id] += ϵ_onsite
        local_measurements["onsite_energy_dn"][orbital_id] += ϵ_onsite
        local_measurements["onsite_energy"][orbital_id] += 2 * ϵ_onsite
    end

    # number of types of hopping
    nhopping = length(tight_binding_parameters.bond_ids)

    # measure hopping energy
    if nhopping > 0
        for hopping_id in 1:nhopping

            # measure bare hopping amplitude
            ϵ_bare_hopping = measure_bare_hopping_energy(greens_estimator, tight_binding_parameters, model_geometry, hopping_id)
            local_measurements["bare_hopping_energy_up"][hopping_id] += ϵ_bare_hopping
            local_measurements["bare_hopping_energy_dn"][hopping_id] += ϵ_bare_hopping
            local_measurements["bare_hopping_energy"][hopping_id] += 2*ϵ_bare_hopping

            # measure hopping energy
            ϵ_hopping = measure_hopping_energy(greens_estimator, fermion_path_integral, tight_binding_parameters, model_geometry, hopping_id)
            local_measurements["bare_hopping_energy_up"][hopping_id] += ϵ_hopping
            local_measurements["bare_hopping_energy_dn"][hopping_id] += ϵ_hopping
            local_measurements["bare_hopping_energy"][hopping_id] += 2*ϵ_hopping

            # measure hopping amplitude
            tbar = measure_hopping_amplitude(tight_binding_parameters, fermion_path_integral, hopping_id)
            local_measurements["hopping_amplitude_up"][hopping_id] += tbar
            local_measurements["hopping_amplitude_dn"][hopping_id] += tbar
            local_measurements["hopping_amplitude"][hopping_id] += tbar

            # measure hopping inversion
            tbar = measure_hopping_inversion(tight_binding_parameters, fermion_path_integral, hopping_id)
            local_measurements["hopping_inversion_up"][hopping_id] += tbar
            local_measurements["hopping_inversion_dn"][hopping_id] += tbar
            local_measurements["hopping_inversion"][hopping_id] += tbar

            # measure hopping inversion
            tbar = measure_hopping_inversion_avg(tight_binding_parameters, fermion_path_integral, hopping_id)
            local_measurements["hopping_inversion_avg_up"][hopping_id] += tbar
            local_measurements["hopping_inversion_avg_dn"][hopping_id] += tbar
            local_measurements["hopping_inversion_avg"][hopping_id] += tbar
        end
    end

    return nothing
end


# measure on-site energy
function measure_onsite_energy(
    greens_estimator::GreensEstimator{E},
    tight_binding_parameters::TightBindingParameters{T,E},
    orbital::Int
) where {E<:AbstractFloat, T<:Number}

    (; GR, Rt, N, L, n, Lτ) = greens_estimator

    # get the chemical poential
    μ = tight_binding_parameters.μ

    # get the on-site energy for each sites
    ϵ = reshape(tight_binding_parameters.ϵ, n, L...)

    # get on-site energies for specified orbital species
    ϵ′ = selectdim(ϵ, 1, orbital)

    # get view into random vector based on orbital index
    GR′ = selectdim(GR, 2, orbital)
    Rt′ = selectdim(Rt, 2, orbital)

    # initialize on-site energy to zero
    e = zero(Complex{E})

    # iterate over unit cells
    for u in CartesianIndices(ϵ′)
        # get relevant views
        GR_u = @view GR′[:, u, :]
        Rt_u = @view Rt′[:, u, :]
        # measure on-site energy for current unit cell
        e += (ϵ′[u] - μ) * sum(1 - GR_u[i] * Rt_u[i] for i in eachindex(Rt_u)) / length(Rt_u)
    end

    # normalize measurement
    e /= N

    return e
end

# measure bare hopping amplitude
function measure_bare_hopping_energy(
    greens_estimator::GreensEstimator{E},
    tight_binding_parameters::TightBindingParameters{T,E},
    model_geometry::ModelGeometry{D,E},
    hopping_id::Int
) where {D, T<:Number, E<:AbstractFloat}

    (; GR, Rt, N, L, n, Lτ, Nrv) = greens_estimator
    (; t, neighbor_table, bond_slices, bond_ids) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(Complex{E})

    # get bond ID associated with hopping ID
    bond_id = bond_ids[hopping_id]

    # get the relevant bond
    bond = model_geometry.bonds[bond_id]

    # get the hopping associated with the bond/hopping in question
    t′ = @view t[bond_slices[hopping_id]]
    t″ = reshape(t′, L)

    # get the relevant orbital species
    b, a = bond.orbitals

    # get shifted view based on bond dispalcement
    r = bond.displacement

    # get the view based on the relevant orbitals species
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)

    # get the shifted view based on the bond displacement to represent
    # cᵀ(i+r) <==> GR(j=i+r) and c(i+r) <==> Rt(j=i+r)
    GR_a_r = ShiftedArrays.circshift(GR_a, (0, (-r[n] for n in 1:D)..., 0))
    Rt_b_r = ShiftedArrays.circshift(Rt_b, (0, (-r[n] for n in 1:D)..., 0))

    # iterate over unit cells
    for i in CartesianIndices(L)
        # get the relevant hopping energy
        ti = t″[i]
        # get the relevant views
        GR_a_i = @view GR_a[:, i, :]
        Rt_b_i = @view Rt_b[:, i, :]
        GR_a_r_i = @view GR_a_r[:, i, :]
        Rt_b_r_i = @view Rt_b_r[:, i, :]
        # calculate hopping energy of current unit cell
        # h =  t⋅cᵀ(i+r)⋅c(i)  + tᵀ⋅cᵀ(i)⋅c(i+r)
        #   = -t⋅c(i)⋅cᵀ(i+r)  - tᵀ⋅c(i+r)⋅cᵀ(i)
        #   ≈ -t⋅GR(i)⋅Rt(i+r) - tᵀ⋅GR(i+r)⋅Rt(i)
        h += sum(
            -ti * GR_a_i[n] * Rt_b_r_i[n] - conj(ti) * GR_a_r_i[n] * Rt_b_i[n]
            for n in eachindex(Rt_b_i) # iterates of imaginary-time slices and random vectors
        ) / (Lτ * Nrv)
    end

    # normalize hopping energy measurement
    h /= N

    return h
end

# measure hopping amplitude
function measure_hopping_energy(
    greens_estimator::GreensEstimator{E},
    fermion_path_integral::FermionPathIntegral{T,E},
    tight_binding_parameters::TightBindingParameters{T,E},
    model_geometry::ModelGeometry{D,E},
    hopping_id::Int
) where {D, T<:Number, E<:AbstractFloat}

    (; GR, Rt, N, L, n, Lτ, Nrv) = greens_estimator
    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; t) = fermion_path_integral

    # initialize hopping energy to zero
    h = zero(Complex{E})

    # get bond ID associated with hopping ID
    bond_id = bond_ids[hopping_id]

    # get the relevant bond
    bond = model_geometry.bonds[bond_id]

    # get the hopping associated with the bond/hopping in question
    t′ = @view t[bond_slices[hopping_id], :]
    t″ = reshape(t′, L..., Lτ)

    # get the relevant orbital species
    b, a = bond.orbitals

    # get shifted view based on bond dispalcement
    r = bond.displacement

    # get the view based on the relevant orbitals species
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)

    # get the shifted view based on the bond displacement to represent
    # cᵀ(i+r) <==> GR(j=i+r) and c(i+r) <==> Rt(j=i+r)
    GR_a_r = ShiftedArrays.circshift(GR_a, (0, (-r[n] for n in 1:D)..., 0))
    Rt_b_r = ShiftedArrays.circshift(Rt_b, (0, (-r[n] for n in 1:D)..., 0))

    # iterate over unit cells
    for i in CartesianIndices(L)
        # get the relevant hopping energy
        ti = @view t″[i, :]
        # get the relevant views
        GR_a_i = @view GR_a[:, i, :]
        Rt_b_i = @view Rt_b[:, i, :]
        GR_a_r_i = @view GR_a_r[:, i, :]
        Rt_b_r_i = @view Rt_b_r[:, i, :]
        # iterate over random vectors
        for rv in 1:Nrv
            # calculate hopping energy of current unit cell
            # h =  t⋅cᵀ(i+r)⋅c(i)  + tᵀ⋅cᵀ(i)⋅c(i+r)
            #   = -t⋅c(i)⋅cᵀ(i+r)  - tᵀ⋅c(i+r)⋅cᵀ(i)
            #   ≈ -t⋅GR(i)⋅Rt(i+r) - tᵀ⋅GR(i+r)⋅Rt(i)
            h += sum(
                -ti[l] * GR_a_i[l, rv] * Rt_b_r_i[l, rv] - conj(ti[l]) * GR_a_r_i[l, rv] * Rt_b_i[l, rv]
                for l in eachindex(ti) # iterates of imaginary-time slices
            ) / (Lτ * Nrv)
        end
    end

    # normalize hopping energy measurement
    h /= N

    return h
end