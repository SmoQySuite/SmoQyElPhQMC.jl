# measure density for single-spin species and specified orbital species
function measure_n(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rt′ = selectdim(greens_estimator.Rt, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    n = measure_n(greens_estimator, Rt′, GR′)

    return n
end

# measure density for single spin-species
function measure_n(
    greens_estimator::GreensEstimator{T},
    Rt::AbstractArray{Complex{T}} = greens_estimator.Rt,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}
    
    R = Rt
    @. R = conj(Rt)
    n = 1 - dot(R, GR)/length(R)
    @. Rt = conj(R)

    return n
end


# measure ⟨N²⟩ globally
function measure_Nsqrd(
    greens_estimator::GreensEstimator{T}
) where {T}

    (; V, N, n, Lτ, Nrv, Rt, GR) = greens_estimator

    # n is number of obritals per unit cell
    # N is number of unit cells
    # Lτ is length of imaginary time axis
    # V is the total space-time volume
    # Nrv is the number of random vectors

    # get original random vectors
    R = Rt
    @. R = conj(Rt)

    # reshape in to matrix where columns correspond to random vectors
    R′ = reshape(R, V, Nrv)
    GR′ = reshape(GR, V, Nrv)

    # measure ⟨N⟩²
    N̄sqrd = zero(Complex{T})
    # iterate over first random vector
    for i in 1:(Nrv-1)
        Ri = @view R′[:,i]
        GRi = @view GR′[:,i]
        # approximate TrG
        TrGi = dot(Ri, GRi)
        # iterate over second random variable
        for j in (i+1):Nrv
            Rj = @view R′[:,j]
            GRj = @view GR′[:,j]
            # approximate TrG
            TrGj = dot(Rj, GRj)
            # approximate ⟨N⟩² = 4⋅(Tr[I]-Tr[G])⋅(Tr[I]-Tr[G])/Lτ²
            # using two indpendent random vectors where Tr[I] = V
            N̄sqrd +=  4 * (V - TrGi) * (V - TrGj) / Lτ^2
        end
    end
    N̄sqrd /= binomial(Nrv, 2)

    # calculate Tr[G]
    TrG = dot(R, GR) / (Nrv * Lτ)

    TrGsqrd = zero(Complex{T})
    # iterate over first random vector
    for i in 1:(Nrv-1)
        Ri = @view R′[:,i]
        GRi = @view GR′[:,i]
        # iterate over second random variable
        for j in (i+1):Nrv
            Rj = @view R′[:,j]
            GRj = @view GR′[:,j]
            TrGsqrd += dot(Rj,GRi) * dot(Ri,GRj) / Lτ^2
        end
    end
    TrGsqrd /= binomial(Nrv, 2)

    Nsqrd = N̄sqrd + 2*TrG/Lτ - 2*TrGsqrd

    # restore conjugated random vectors
    @. Rt = conj(R)

    return Nsqrd
end


# measure the double-occupancy for specific spin species
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rt′ = selectdim(greens_estimator.Rt, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    d = measure_double_occ(greens_estimator, Rt′, GR′)

    return d
end

# measure the globally averaged double-occupancy
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    Rt::AbstractArray{Complex{T}} = greens_estimator.Rt,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}

    (; V) = greens_estimator

    # number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # number of random vector pairs
    Npairs = binomial(Nrv, 2)

    # initialize double-occupancy to zero
    d = zero(Complex{T})

    # iterate over all pairs of random vectors
    for i in 1:(Nrv-1)
        GRup = selectdim(GR, ndims(GR), i)
        Rtup = selectdim(Rt, ndims(Rt), i)
        for j in (i+1):Nrv
            GRdn = selectdim(GR, ndims(GR), j)
            Rtdn = selectdim(Rt, ndims(Rt), j)
            # ⟨n₊n₋⟩ = ⟨(1-G₊(0))⋅(1-G₋(0))⟩
            d += sum(
                r -> (1-GRup[r]*Rtup[r]) * (1-GRdn[r]*Rtdn[r]),
                eachindex(Rtup)
            ) / V
        end
    end

    # normalize double occupancy measurement
    d /= Npairs

    return d
end