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

    (; V, N, n, Lτ, Nrv, Rt, GR, CΔ0, pfft!, pifft!) = greens_estimator
    Gl = greens_estimator.A
    Gr = greens_estimator.B

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

    # restore conjugated random vectors
    @. Rt = conj(R)

    # measure Tr[G²]
    TrGsqrd = zero(Complex{T})
    # iterate over pairs of orbitals
    for a in 1:n
        for b in 1:n
            GR_a = selectdim(GR, 2, a)
            Rt_b = selectdim(Rt, 2, b)
            GR_b = selectdim(GR, 2, b)
            Rt_a = selectdim(Rt, 2, a)
            fill!(CΔ0, 0)
            # iterate over pairs of random vectors
            for i in 1:(Nrv-1)
                for j in (i+1):Nrv
                    # get views for appropriate random vectors
                    GR_a_i = selectdim(GR_a, ndims(GR_a), i)
                    Rt_b_i = selectdim(Rt_b, ndims(Rt_b), i)
                    GR_b_j = selectdim(GR_b, ndims(GR_b), j)
                    Rt_a_j = selectdim(Rt_a, ndims(Rt_a), j)
                    # calculate G(0,τ)⋅G(τ,0)
                    _measure_CΔ0!(
                        CΔ0, Gl, Gr,
                        GR_b_j, Rt_b_i, Rt_a_j, GR_a_i,
                        pfft!, pifft!
                    )
                end
            end
            G0G0 = selectdim(CΔ0, 1, 1)
            TrGsqrd += N * sum(G0G0) / binomial(Nrv, 2)
        end
    end

    # calculate ⟨N²⟩
    Nsqrd = N̄sqrd + 2*TrG - 2*TrGsqrd

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