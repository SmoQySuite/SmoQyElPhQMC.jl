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

    (; N, n, Lτ, Nrv) = greens_estimator
    (; bond_slices) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(Complex{E})

    # get the hopping associated with the hopping id
    t = @view tight_binding_parameters.t[bond_slices[hopping_id]]

    # get the neighbor table associated with the hopping id
    neighbor_table = @view tight_binding_parameters.neighbor_table[:,bond_slices[hopping_id]]

    # total number of sites in lattice given that N is number of unit cells
    # and n is the number of orbitals per unit cell
    Nsites = N * n

    # get random vector view
    GR = reshape(greens_estimator.GR, (Lτ, Nsites, Nrv))
    Rt = reshape(greens_estimator.Rt, (Lτ, Nsites, Nrv))

    # iterate over random vector
    @inbounds for rv in 1:Nrv
        # iterate over hoppings
        for m in 1:N
            # get the pair of sites connected by the hopping
            i = neighbor_table[1,m] # initial site
            f = neighbor_table[2,m] # final site
            # get the hopping amplitude
            t_if = t[m]
            # iterate over imaginary-time slice
            for l in 1:Lτ
                # calculate hopping energy of current unit cell
                # h = -t⋅cᵀ(i+r)⋅c(i)  - tᵀ⋅cᵀ(i)⋅c(i+r)
                #   = +t⋅c(i)⋅cᵀ(i+r)  + tᵀ⋅c(i+r)⋅cᵀ(i)
                #   ≈ +t⋅GR(i)⋅Rt(i+r) + tᵀ⋅GR(i+r)⋅Rt(i)
                a = t_if * GR[l,i,rv] * Rt[l,f,rv]
                b = conj(t_if) * GR[l,f,rv] * Rt[l,i,rv]
                h += a + b
            end
        end
    end

    # normalize hopping energy measurement
    h /= (Lτ * Nsites * Nrv)

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

    (; N, n, Lτ, Nrv) = greens_estimator
    (; bond_slices) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(Complex{E})

    # get the hopping associated with the hopping id
    t = @view fermion_path_integral.t[bond_slices[hopping_id],:]

    # get the neighbor table associated with the hopping id
    neighbor_table = @view tight_binding_parameters.neighbor_table[:,bond_slices[hopping_id]]

    # total number of sites in lattice given that N is number of unit cells
    # and n is the number of orbitals per unit cell
    Nsites = N * n

    # get random vector view
    GR = reshape(greens_estimator.GR, (Lτ, Nsites, Nrv))
    Rt = reshape(greens_estimator.Rt, (Lτ, Nsites, Nrv))

    # iterate over random vector
    @inbounds for rv in 1:Nrv
        # iterate over hoppings
        for m in 1:N
            # get the pair of sites connected by the hopping
            i = neighbor_table[1,m] # initial site
            f = neighbor_table[2,m] # final site
            # iterate over imaginary-time slice
            for l in 1:Lτ
                # get the hopping amplitude
                t_if = t[m,l]
                # calculate hopping energy of current unit cell
                # h = -t⋅cᵀ(i+r)⋅c(i)  - tᵀ⋅cᵀ(i)⋅c(i+r)
                #   = +t⋅c(i)⋅cᵀ(i+r)  + tᵀ⋅c(i+r)⋅cᵀ(i)
                #   ≈ +t⋅GR(i)⋅Rt(i+r) + tᵀ⋅GR(i+r)⋅Rt(i)
                a = t_if * GR[l,i,rv] * Rt[l,f,rv]
                b = conj(t_if) * GR[l,f,rv] * Rt[l,i,rv]
                h += a + b
            end
        end
    end

    # normalize hopping energy measurement
    h /= (Lτ * Nsites * Nrv)

    return h
end