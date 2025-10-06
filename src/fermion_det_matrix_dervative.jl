# calculate ν⋅Re[⟨u|∂M/∂x|v⟩]
function mul_νRe∂M∂x!(
    νRe∂M∂x::AbstractMatrix{E},
    ν::E,
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    fermion_det_matrix::SymFermionDetMatrix{T,E},
    elph::ElectronPhononParameters{T,E},
) where {T<:Number, E<:AbstractFloat}

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table,
       checkerboard_colors, checkerboard_perm) = fermion_det_matrix
    v′, u′ = fermion_det_matrix.tmp1, fermion_det_matrix.tmp2
    Δτ = elph.Δτ
    ssh_parameters = elph.ssh_parameters_up
    holstein_parameters = elph.holstein_parameters_up
    (; Nssh) = ssh_parameters
    (; Nholstein) = holstein_parameters

    u = reshaped(u, size(u′))
    v = reshaped(v, size(v′))

    # number of checkerboard colors
    Ncolors = length(checkerboard_colors)

    # v′[l] = v[l-1]
    circshift!(v′, v, (1,0))

    # v′[l] = -v′[l] = -v[l-1] for l>1
    @views @. v′[2:end,:] = -v′[2:end,:]

    # caluculate v′[l] = exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    checkerboard_lmul!(v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt, transposed = true)

    # calculate v′[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    @. v′ = expnΔτV * v′

    # calculate v′[l] = B[l]⋅v[l-1] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]
    checkerboard_lmul!(v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt, transposed = false)

    # u′[l] = u[l]
    copyto!(u′, u)

    # First, evaluate terms from ⟨u[l]|[∂exp(-Δτ⋅K[l]/2)/dx]⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ|v[l-1]⟩
    # At this point |u′[l]⟩ = |u[l]⟩ and |v′[l]⟩ = B[l]|v[l-1]⟩ = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ⋅v[l-1]⟩.

    # check if there are any ssh couplings coupling
    if Nssh > 0
        # iterate over checkerboard colors in reverse order
        for color in Ncolors:-1:1
            # calculate -ν⋅Re[⟨u′|Δτ⋅∂Kc/∂x|v′⟩]
            _mul_νReΔτ∂Kc∂x!(νRe∂M∂x, -ν, u′, v′, fermion_det_matrix, elph, Δτ/2, color)
            # |u′⟩ := exp(-Δτ⋅Kc)|u′⟩
            checkerboard_lmul!(
                u′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = false, interval = checkerboard_colors[color]
            )
            # |v′⟩ := exp(+Δτ⋅Kc)|v′⟩
            checkerboard_ldiv!(
                v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = false, interval = checkerboard_colors[color]
            )
        end
    else
        # |u′⟩ := exp(-Δτ⋅K)ᵀ|u′⟩
        checkerboard_lmul!(
            u′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
            transposed = true
        )
        # |v′⟩ := exp(+Δτ⋅K)ᵀ|v′⟩
        checkerboard_ldiv!(
            v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
            transposed = true
        )
    end

    # Second, evaluate terms from ⟨u[l]|exp(-Δτ⋅K[l]/2)⋅[∂exp(-Δτ⋅V[l])/∂x]⋅exp(-Δτ⋅K[l]/2)ᵀ|v[l-1]⟩.
    # At this point |u′[l]⟩ = exp(-Δτ⋅K[l]/2)⋅|u[l]⟩ and |v′[l]⟩ = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2)ᵀ|v[l-1]⟩.

    # check if there are any holstein coupling
    if Nholstein > 0
        # calculate -ν⋅Re[⟨u′|Δτ⋅∂V/∂x|v′⟩]
        _mul_νReΔτ∂V∂x!(νRe∂M∂x, -ν, u′, v′, elph)
    end

    # |u′[l]⟩ := exp(-Δτ⋅V)|u′[l]⟩ so that |u′[l]⟩ = exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ⋅|u[l]⟩
    @. u′ *= expnΔτV

    # |v′[l]⟩ := exp(+Δτ⋅V)|v′[l]⟩ so that |v′[l]⟩ = exp(-Δτ⋅K/2)ᵀ|v[l-1]⟩ now
    @. v′ *= inv(expnΔτV)

    # Lastly, evaluate terms from ⟨u[l]|exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅[∂exp(-Δτ⋅K[l]/2)ᵀ/∂x]|v[l-1]⟩.

    # check if there are any ssh couplings coupling
    if Nssh > 0
        # iterate over checkerboard colors in reverse order
        for color in 1:Ncolors
            # calculate -ν⋅Re[⟨u′|Δτ⋅∂Kc/∂x|v′⟩]
            _mul_νReΔτ∂Kc∂x!(νRe∂M∂x, -ν, u′, v′, fermion_det_matrix, elph, Δτ/2, color)
            # |u′⟩ := exp(-Δτ⋅Kc)|u′⟩
            checkerboard_lmul!(
                u′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = false, interval = checkerboard_colors[color]
            )
            # |v′⟩ := exp(+Δτ⋅Kc)|v′⟩
            checkerboard_ldiv!(
                v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = false, interval = checkerboard_colors[color]
            )
        end
    end

    return nothing
end

# calculate ν⋅⟨u|Re(∂M/∂x)|v⟩
function mul_νRe∂M∂x!(
    νRe∂M∂x::AbstractMatrix{E},
    ν::E,
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    fermion_det_matrix::AsymFermionDetMatrix{T,E},
    elph::ElectronPhononParameters{T,E}
) where {T<:Number, E<:AbstractFloat}

    (; expnΔτV, coshΔτt, sinhΔτt, checkerboard_neighbor_table,
       checkerboard_colors, checkerboard_perm) = fermion_det_matrix
    v′, u′ = fermion_det_matrix.tmp1, fermion_det_matrix.tmp2
    Δτ = elph.Δτ
    ssh_parameters = elph.ssh_parameters_up
    holstein_parameters = elph.holstein_parameters_up
    (; Nssh) = ssh_parameters
    (; Nholstein) = holstein_parameters

    u = reshaped(u, size(u′))
    v = reshaped(v, size(v′))

    # number of checkerboard colors
    Ncolors = length(checkerboard_colors)

    # v′[l] = v[l-1]
    circshift!(v′, v, (1,0))

    # v′[l] = -v′[l] = -v[l-1] for l>1
    @views @. v′[2:end,:] = -v′[2:end,:]

    # caluculate v′[l] = exp(-Δτ⋅K[l])⋅v[l-1]
    checkerboard_lmul!(v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt, transposed = false)

    # calculate v′[l] = B[l]⋅v[l-1] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])⋅v[l-1]
    @. v′ = expnΔτV * v′

    # u′[l] = u[l]
    copyto!(u′, u)

    # Second, evaluate terms from ⟨u[l]|[∂exp(-Δτ⋅V[l])/∂x]⋅exp(-Δτ⋅K[l])|v[l-1]⟩.
    # At this point |u′[l]⟩ = |u[l]⟩ and |v′[l]⟩ = B[l]|v[l-1]⟩ = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])|v[l-1]⟩.

    # check if there are any holstein coupling
    if Nholstein > 0
        # calculate -ν⋅Re[⟨u′|Δτ⋅∂V/∂x|v′⟩]
        _mul_νReΔτ∂V∂x!(νRe∂M∂x, -ν, u′, v′, elph)
    end

    # Lastly, evaluate terms from ⟨u[l]|exp(-Δτ⋅V[l])⋅[∂exp(-Δτ⋅K[l])/∂x]|v[l-1]⟩.

    # check if there are any ssh couplings coupling
    if Nssh > 0
        # |u′[l]⟩ := exp(-Δτ⋅V)|u′[l]⟩ so that |u′[l]⟩ = exp(-Δτ⋅V[l])|u[l]⟩
        @. u′ = expnΔτV * u′
        # |v′[l]⟩ := exp(+Δτ⋅V)|v′[l]⟩ so that |v′[l]⟩ = exp(-Δτ⋅K[l])|v[l-1]⟩ now
        @. v′ = v′ / expnΔτV
        # iterate over checkerboard colors in reverse order
        for color in Ncolors:-1:1
            # calculate -ν⋅Re[⟨u′|Δτ⋅∂Kc/∂x|v′⟩]
            _mul_νReΔτ∂Kc∂x!(νRe∂M∂x, -ν, u′, v′, fermion_det_matrix, elph, Δτ, color)
            # |u′⟩ := exp(-Δτ⋅Kc)ᵀ|u′⟩ = exp(-Δτ⋅Kc)|u′⟩
            checkerboard_lmul!(
                u′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = false, interval = checkerboard_colors[color]
            )
            # |v′⟩ := exp(+Δτ⋅Kc)|v′⟩
            checkerboard_ldiv!(
                v′, checkerboard_neighbor_table, coshΔτt, sinhΔτt,
                transposed = true, interval = checkerboard_colors[color]
            )
        end
    end

    return nothing
end


# calculate ν⋅⟨u′|Re(Δτ⋅∂Kc/∂x)|v′⟩ where Kc is the kinetic energy matrix associated with
# one of the checkerboard colors
function _mul_νReΔτ∂Kc∂x!(
    νRe∂M∂x::AbstractMatrix{E},
    ν::E,
    u′::AbstractMatrix,
    v′::AbstractMatrix,
    fermion_det_matrix::FermionDetMatrix{T,E},
    elph::ElectronPhononParameters{T,E},
    Δτ::E,
    color::Int
) where {T<:Number, E<:AbstractFloat}

    (; checkerboard_neighbor_table, checkerboard_colors, checkerboard_perm) = fermion_det_matrix
    (; x, phonon_parameters) = elph
    ssh_parameters = elph.ssh_parameters_up
    (; α, α2, α3, α4, coupling_to_phonon, hopping_to_couplings) = ssh_parameters
    (; M) = phonon_parameters

    # iterate over hoppings in current checkerboard color
    for n in checkerboard_colors[color]
        # get hopping index
        h = checkerboard_perm[n]
        # get hopping to couplings map
        h_to_c = hopping_to_couplings[h]
        # check if there are any ssh couplings to current hopping
        if !isempty(h_to_c)
            # iterate over ssh coupling associated with hopping
            for c in h_to_c
                # get the pair of phonons getting coupled
                p  = coupling_to_phonon[1,c]
                p′ = coupling_to_phonon[2,c]
                # get the pair of orbitals that the coupled phonons live on
                i = checkerboard_neighbor_table[1,n]
                j = checkerboard_neighbor_table[2,n]
                # iterate over imaginary time slices
                for l in axes(x,2)
                    # calculate relative phonon position (x′ - x)
                    Δx = x[p′,l] - x[p,l]
                    # if mass of phonon p is finite
                    if isfinite(M[p])
                        # calculate Δτ⋅∂Kc/∂x[j,i]
                        ΔτdKcdx_ji = Δτ * (-α[c] - 2*α2[c]*Δx - 3*α3[c]*Δx^2 - 4*α4[c]*Δx^3)
                        # calculate ν⋅Re[⟨u′|Δτ⋅∂Kc/∂x|v′⟩]
                        νRe∂M∂x[p,l] += ν * real( conj(u′[l,j]) * ΔτdKcdx_ji * v′[l,i] + conj(u′[l,i]) * conj(ΔτdKcdx_ji) * v′[l,j] )
                    end
                    # if mass of phonon p′ is finite
                    if isfinite(M[p′])
                        # calculate Δτ⋅∂Kc/∂x′[j,i]
                        ΔτdKcdx_ji = Δτ * (α[c] + 2*α2[c]*Δx + 3*α3[c]*Δx^2 + 4*α4[c]*Δx^3)
                        # calculate ν⋅Re[⟨u′|Δτ⋅∂Kc/∂x|v′⟩]
                        νRe∂M∂x[p′,l] += ν * real( conj(u′[l,j]) * ΔτdKcdx_ji * v′[l,i] + conj(u′[l,i]) * conj(ΔτdKcdx_ji) * v′[l,j] )
                    end
                end
            end
        end
    end

    return nothing
end


# calculate ν⋅Re[⟨u′|Δτ⋅∂V/∂x|v′⟩]
function _mul_νReΔτ∂V∂x!(
    ν∂M∂x::AbstractMatrix{E},
    ν::E,
    u′::AbstractMatrix,
    v′::AbstractMatrix,
    elph::ElectronPhononParameters{T,E},
) where {T<:Number, E<:AbstractFloat}

    (; β, Δτ, x, phonon_parameters) = elph
    holstein_parameters = elph.holstein_parameters_up
    (; α, α2, α3, α4, coupling_to_phonon, neighbor_table, Nholstein) = holstein_parameters
    (; M) = phonon_parameters

    # iterate over holstein couplings
    for c in 1:Nholstein
        # get the phonon associated with the coupling
        p = coupling_to_phonon[c]
        # get the orbital whose density is getting coupled to
        i = neighbor_table[2,c]
        # if phonon mass is finite
        if isfinite(M[p])
            # iterate over imaginary time slice
            for l in axes(x,2)
                # calculate Δτ⋅∂V/∂x
                ΔτdVdx = Δτ * (α[c] + 2*α2[c]*x[p,l] + 3*α3[c]*x[p,l]^2 + 4*α4[c]*x[p,l]^3)
                # calculate ν⋅Re[⟨u′|Δτ⋅∂V/∂x|v′⟩]
                ν∂M∂x[p,l] += ν * real(conj(u′[l,i]) * ΔτdVdx * v′[l,i])
            end
        end
    end

    return nothing
end