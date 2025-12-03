# make electron-phonon measurements
function make_electron_phonon_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    model_geometry::ModelGeometry{D,E,N},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    greens_estimator::GreensEstimator{E},
) where {E<:AbstractFloat, T<:Number, D, N}

    x = electron_phonon_parameters.x
    nphonon = electron_phonon_parameters.phonon_parameters.nphonon::Int # number of phonon modes per unit cell
    nholstein = electron_phonon_parameters.holstein_parameters_up.nholstein::Int # number of types of holstein couplings
    nssh = electron_phonon_parameters.ssh_parameters_up.nssh::Int # number of types of ssh coupling
    ndispersion = electron_phonon_parameters.dispersion_parameters.ndispersion::Int # number of types of dispersive phonon couplings

    # make phonon mode related measurements
    for phonon_id in 1:nphonon
        local_measurements["phonon_kin_energy"][phonon_id]   += measure_phonon_kinetic_energy(electron_phonon_parameters, phonon_id)
        local_measurements["phonon_pot_energy"][phonon_id] += measure_phonon_potential_energy(electron_phonon_parameters, phonon_id)
        local_measurements["X"][phonon_id]  += measure_phonon_position_moment(electron_phonon_parameters, phonon_id, 1)
        local_measurements["X2"][phonon_id] += measure_phonon_position_moment(electron_phonon_parameters, phonon_id, 2)
        local_measurements["X3"][phonon_id] += measure_phonon_position_moment(electron_phonon_parameters, phonon_id, 3)
        local_measurements["X4"][phonon_id] += measure_phonon_position_moment(electron_phonon_parameters, phonon_id, 4)
    end

    # check if finite number of holstein couplings
    if nholstein > 0
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        # make holstein coupling related measurements
        for holstein_id in 1:nholstein
            ϵ_hol = measure_holstein_energy(holstein_parameters, greens_estimator, x, holstein_id)
            local_measurements["holstein_energy_up"][holstein_id] += ϵ_hol
            local_measurements["holstein_energy_dn"][holstein_id] += ϵ_hol
            local_measurements["holstein_energy"][holstein_id] += 2 * ϵ_hol
        end
    end

    # check if finite number of ssh couplings
    if nssh > 0
        # make ssh coupling related measurements
        for ssh_id in 1:nssh
            ssh_parameters = electron_phonon_parameters.ssh_parameters_up
            ϵ_ssh = measure_ssh_energy(ssh_parameters, greens_estimator, x, ssh_id)
            local_measurements["ssh_energy_up"][ssh_id] += ϵ_ssh
            local_measurements["ssh_energy_dn"][ssh_id] += ϵ_ssh
            local_measurements["ssh_energy"][ssh_id] += 2 * ϵ_ssh
        end
    end

    # check if finite number of dispersive phonon coupling
    if ndispersion > 0
        # make ssh coupling related measurements
        for dispersion_id in 1:ndispersion
            local_measurements["dispersion_energy"][dispersion_id] += measure_dispersion_energy(electron_phonon_parameters, dispersion_id)
        end
    end

    return nothing
end

# measure holstein interaction energy
function measure_holstein_energy(
    holstein_parameters::HolsteinParameters{E},
    greens_estimator::GreensEstimator{E, D},
    x::Matrix{E},
    holstein_id::Int
) where {D, E<:AbstractFloat}

    (; nholstein, Nholstein, coupling_to_site, coupling_to_phonon, ph_sym_form) = holstein_parameters
    (; Nrv, L, N, n, Lτ) = greens_estimator

    # initialize holstein electron-phonon coupling energy to zero
    ϵ_hol = zero(Complex{E})

    # if using particle-hole symmetric definition
    phs = ph_sym_form[holstein_id]

    # get phonon fields associated with coupling ID
    phonon_i = coupling_to_phonon[(holstein_id-1) * N + 1]
    phonon_f = coupling_to_phonon[holstein_id * N]
    x′ = @view x[phonon_i:phonon_f, :]
    x″ = reshape(x′, (L..., Lτ))

    # determine the orbital species of the orbital density that the phonon
    # is coupling to for this interaction
    coupling = (holstein_id-1)*N+1
    site = coupling_to_site[coupling]
    orbital_id = mod1(site, n)

    # get views based on orbital ID
    GR = selectdim(greens_estimator.GR, 2, orbital_id)
    Rt = selectdim(greens_estimator.Rt, 2, orbital_id)

    # get coupling parameters
    α1_all = reshape(holstein_parameters.α, L..., nholstein)
    α2_all = reshape(holstein_parameters.α2, L..., nholstein)
    α3_all = reshape(holstein_parameters.α3, L..., nholstein)
    α4_all = reshape(holstein_parameters.α4, L..., nholstein)

    # get coupling parameters for specified holstein ID
    α1 = selectdim(α1_all, D+1, holstein_id)
    α2 = selectdim(α2_all, D+1, holstein_id)
    α3 = selectdim(α3_all, D+1, holstein_id)
    α4 = selectdim(α4_all, D+1, holstein_id)

    # iterate over unit cells
    for i in CartesianIndices(L)
        # iterate over imaginary-time slices
        for l in 1:Lτ
            # estimate the density using all random vectors
            n_li = sum(1.0 - GR[l, i, rv] * Rt[l, i, rv] for rv in 1:Nrv) / Nrv
            # get phonon field
            x_il = x″[i, l]
            # calculate holstein interaction energy
            ϵ_hol += (α2[i]*x_il^2 + α4[i]*x_il^4) * n_li
            ϵ_hol += phs ? (α1[i]*x_il^1 + α3[i]*x_il^2) * (n_li - 0.5) : (α1[i]*x_il^1 + α3[i]*x_il^2) * n_li
        end
    end

    # normalize measurement
    ϵ_hol /= (N * Lτ)

    return ϵ_hol
end

# measure holstein interaction energy
function measure_ssh_energy(
    ssh_parameters::SSHParameters{T},
    greens_estimator::GreensEstimator{E, D},
    x::Matrix{E},
    ssh_id::Int
) where {D, T<:Number, E<:AbstractFloat}

    (; nssh, coupling_to_phonon) = ssh_parameters
    (; Nrv, L, N, n, Lτ) = greens_estimator

    # initialize ssh energy to zero
    ϵ_ssh = zero(Complex{E})

    # get the relevant view into neighbor table and phonon mapping
    ssh_index_i = (ssh_id-1) * N + 1
    ssh_index_f = ssh_id * N
    slice = ssh_index_i:ssh_index_f
    neighbor_table = @view ssh_parameters.neighbor_table[:, slice]
    coupling_to_phonon = @view ssh_parameters.coupling_to_phonon[:, slice]

    # get views based on orbital ID
    GR = reshape(greens_estimator.GR, Lτ, N*n, Nrv)
    Rt = reshape(greens_estimator.Rt, Lτ, N*n, Nrv)

    # get coupling parameters
    α1_all = reshape(ssh_parameters.α, N, nssh)
    α2_all = reshape(ssh_parameters.α2, N, nssh)
    α3_all = reshape(ssh_parameters.α3, N, nssh)
    α4_all = reshape(ssh_parameters.α4, N, nssh)

    # get coupling parameters for specified holstein ID
    α1 = selectdim(α1_all, 2, ssh_id)
    α2 = selectdim(α2_all, 2, ssh_id)
    α3 = selectdim(α3_all, 2, ssh_id)
    α4 = selectdim(α4_all, 2, ssh_id)

    # iterate over unit cells
    for u in 1:N
        # get pair of sites
        s_i, s_f = neighbor_table[1,u], neighbor_table[2,u]
        # get pair of phonon modes
        p_i, p_f = coupling_to_phonon[1,u], coupling_to_phonon[2,u]
        # iterate over imaginary-time slices
        for l in 1:Lτ
            # get relative phonon displacement
            Δx = x[p_f, l] - x[p_i, l]
            # calculate coupling
            c_ul = α1[u] * Δx + α2[u] * Δx^2 + α3[u] * Δx^3 + α4[u] * Δx^4
            # get forward hopping amplitude
            hf = -sum(GR[l, s_i, rv] * Rt[l, s_f, rv] for rv in 1:Nrv)/Nrv
            # get reverse hopping process
            hr = -sum(GR[l, s_f, rv] * Rt[l, s_i, rv] for rv in 1:Nrv)/Nrv
            # calculate interaction energy
            ϵ_ssh += c_ul * hf + conj(c_ul) * hr
        end
    end

    # normalize measurement
    ϵ_ssh /= (N * Lτ)

    return ϵ_ssh
end