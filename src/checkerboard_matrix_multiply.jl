# perform matrix-vector multiply with checkerboard matrix
function checkerboard_mul!(
    v′::AbstractVecOrMat{T},
    v::AbstractVecOrMat{T},
    neighbor_table::Matrix{Int},
    coshΔτt::Matrix{E},
    sinhΔτt::Matrix{E};
    transposed::Bool = false,
    interval::UnitRange{Int} = 1:size(neighbor_table,2)
) where {T<:Number, E<:Number}

    # copy vector over
    copyto!(v′, v)
    
    # perform checkerboard multiply
    checkerboard_lmul!(
        v′, neighbor_table, coshΔτt, sinhΔτt,
        transposed = transposed, interval = interval
    )

    return nothing
end


# checkerboard left multiply
function checkerboard_lmul!(
    v::AbstractVecOrMat{T},
    neighbor_table::Matrix{Int},
    coshΔτt::Matrix{E},
    sinhΔτt::Matrix{E};
    transposed::Bool = false,
    interval::UnitRange{Int} = 1:size(neighbor_table,2)
) where {T<:Number, E<:Number}

    # number of imaginary-time slice
    Lτ = size(coshΔτt, 1)
    
    # number of orbitals in lattice
    N = length(v) ÷ Lτ

    # reshaped if vector to matrix
    u = (ndims(v) == 2) ? v : reshaped(v, (Lτ, N))

    # if transposed, then reverse the order the hoppings are iterated over
    if transposed
        interval = reverse(interval)
    end

    # iterate over specified sequence of hoppings
    @inbounds for h in interval
        # get the pair of orbitals connected by hopping h
        i = neighbor_table[1, h]
        j = neighbor_table[2, h]
        # construct relevant views
        u_i = @view u[:,i]
        u_j = @view u[:,j]
        c = @view coshΔτt[:,h]
        s = @view sinhΔτt[:,h]
        # iterate over imaginary-time slice
        @simd for l in eachindex(u_i)
            # update vector elements
            u_li = u_i[l]
            u_lj = u_j[l]
            c_ij = c[l]
            s_ij = s[l]
            u_i[l] = c_ij * u_li + s_ij * u_lj
            u_j[l] = c_ij * u_lj + conj(s_ij) * u_li
        end
    end

    return nothing
end


# left multiply by inverse of checkerboard matrix
function checkerboard_ldiv!(
    v′::AbstractVecOrMat{T},
    v::AbstractVecOrMat{T},
    neighbor_table::Matrix{Int},
    coshΔτt::Matrix{E},
    sinhΔτt::Matrix{E};
    transposed::Bool = false,
    interval::UnitRange{Int} = 1:size(neighbor_table,2)
) where {T<:Number, E<:Number}

    # copy vector
    copyto!(v′, v)

    # perform checkerboard multiply
    checkerboard_ldiv!(
        v′, neighbor_table, coshΔτt, sinhΔτt,
        transposed = transposed, interval = interval
    )

    return nothing
end

# left multiply by inverse of checkerboard matrix
function checkerboard_ldiv!(
    v::AbstractVecOrMat{T},
    neighbor_table::Matrix{Int},
    coshΔτt::Matrix{E},
    sinhΔτt::Matrix{E};
    transposed::Bool = false,
    interval::UnitRange{Int} = 1:size(neighbor_table,2)
) where {T<:Number, E<:Number}

    # number of imaginary-time slice
    Lτ = size(coshΔτt, 1)
    
    # number of orbitals in lattice
    N = length(v) ÷ Lτ

    # reshaped if vector to matrix
    u = (ndims(v) == 2) ? v : reshaped(v, (Lτ, N))

    # if not transposed, then reverse the order the hoppings are iterated over
    if !transposed
        interval = reverse(interval)
    end

    # iterate over specified sequence of hoppings
    @inbounds for h in interval
        # get the pair of orbitals connected by hopping h
        i = neighbor_table[1, h]
        j = neighbor_table[2, h]
        # construct views
        u_i = @view u[:,i]
        u_j = @view u[:,j]
        c = @view coshΔτt[:,h]
        s = @view sinhΔτt[:,h]
        # iterate over imaginary-time slice
        @simd for l in eachindex(u_i)
            # update vector elements
            u_li = u_i[l]
            u_lj = u_j[l]
            c_ij = c[l]
            s_ij = s[l]
            u_i[l] = c_ij * u_li - s_ij * u_lj
            u_j[l] = c_ij * u_lj - conj(s_ij) * u_li
        end
    end

    return nothing
end