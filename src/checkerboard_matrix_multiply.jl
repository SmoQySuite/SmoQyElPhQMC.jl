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
    u = reshaped(v, Lτ, N)

    # if transposed, then reverse the order the hoppings are iterated over
    if transposed
        interval = reverse(interval)
    end

    # iterate over specified sequence of hoppings
    for h in interval
        # get the pair of orbitals connected by hopping h
        i = neighbor_table[1, h]
        j = neighbor_table[2, h]
        # iterate over imaginary-time slice
        @simd for l in axes(u, 1)
            # get relevant cosh and sinh values
            c_ij = coshΔτt[l, h]
            s_ij = sinhΔτt[l, h]
            # get initial vector values that will be modified
            u_i = u[l,i]
            u_j = u[l,j]
            # update vector elements
            u[l,i] = c_ij * u_i + s_ij * u_j
            u[l,j] = c_ij * u_j + conj(s_ij) * u_i
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
    u = reshaped(v, Lτ, N)

    # if not transposed, then reverse the order the hoppings are iterated over
    if !transposed
        interval = reverse(interval)
    end

    # iterate over specified sequence of hoppings
    for h in interval
        # get the pair of orbitals connected by hopping h
        i = neighbor_table[1, h]
        j = neighbor_table[2, h]
        # iterate over imaginary-time slice
        @simd for l in axes(u, 1)
            # get relevant cosh and sinh values
            c_ij = coshΔτt[l, h]
            s_ij = sinhΔτt[l, h]
            # get initial vector values that will be modified
            u_i = u[l,i]
            u_j = u[l,j]
            # update vector elements
            u[l,i] = c_ij * u_i - s_ij * u_j
            u[l,j] = c_ij * u_j - conj(s_ij) * u_i
        end
    end

    return nothing
end